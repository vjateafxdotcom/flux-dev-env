#!/usr/bin/env python3
"""MCP Image Generation Server for newbot.lan (3x RTX PRO 4000 Blackwell)
Enhanced Version: Preserves resolution and maximizes realism.
"""
import os, sys, time, base64, io, gc, asyncio, argparse, numpy as np, functools
from pathlib import Path
from typing import Optional
import torch
from PIL import Image, ImageOps, ImageFilter
from diffusers import QwenImagePipeline, FluxPipeline, FluxImg2ImgPipeline, FluxInpaintPipeline
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from mcp.server.fastmcp import FastMCP

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("HF_TOKEN", "")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

MODEL_ID = "Qwen/Qwen-Image"
FLUX_DEV_SNAPSHOT = "/home/eafxadministrator/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
CLIPSEG_MODEL = "CIDAS/clipseg-rd64-refined"
OUTPUT_DIR = Path("/home/eafxadministrator/flux-dev-env/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_pipe = None
_flux_pipe = None
_mask_model = None
_mask_processor = None

mcp = FastMCP("MCP Image Generation Server (Enhanced)")

def _get_clipseg():
    global _mask_model, _mask_processor
    if _mask_model is not None: return _mask_model, _mask_processor
    _mask_processor = CLIPSegProcessor.from_pretrained(CLIPSEG_MODEL)
    _mask_model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_MODEL)
    if torch.cuda.is_available(): _mask_model = _mask_model.to("cuda")
    return _mask_model, _mask_processor

def _get_flux_pipe():
    global _flux_pipe
    if _flux_pipe is not None: return _flux_pipe
    try:
        _flux_pipe = FluxInpaintPipeline.from_pretrained(FLUX_DEV_SNAPSHOT, torch_dtype=torch.float16, local_files_only=True)
        _flux_pipe.enable_sequential_cpu_offload()
    except Exception:
        _flux_pipe = FluxInpaintPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
    return _flux_pipe

def _get_pipe():
    global _pipe
    if _pipe is not None: return _pipe
    _pipe = QwenImagePipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    if torch.cuda.is_available(): _pipe.enable_sequential_cpu_offload()
    return _pipe

def _img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def _b64_to_img(b64_str):
    if isinstance(b64_str, str) and b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")

def _resize_for_flux(img):
    w, h = img.size
    max_dim = 1536
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        w, h = int(w * scale), int(h * scale)
    w = (w // 32) * 32
    h = (h // 32) * 32
    return img.resize((w, h), Image.LANCZOS)

def _segment_object_sync(image_b64, text):
    pipe_mask, proc = _get_clipseg()
    image = _b64_to_img(image_b64)
    inputs = proc(text=text, images=[image], return_tensors="pt")
    for k, v in inputs.items():
        if hasattr(v, "to"): inputs[k] = v.to(pipe_mask.device)
    with torch.no_grad():
        outputs = pipe_mask(**inputs)
    mask_np = torch.sigmoid(outputs.logits[0]).cpu().numpy()
    try:
        from skimage.filters import threshold_otsu
        thresh = threshold_otsu(mask_np)
        mask_bool = mask_np > thresh
    except ImportError:
        mask_bool = mask_np > 0.5
    from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure
    struct = generate_binary_structure(2, 1)
    mask_bool = binary_opening(mask_bool, structure=struct)
    mask_bool = binary_closing(mask_bool, structure=struct)
    mask_img = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode='L')
    if mask_img.size != image.size:
        mask_img = mask_img.resize(image.size, Image.LANCZOS)
    return _img_to_b64(mask_img)

@mcp.tool(description="Generate high-quality image from text prompt.")
async def generate_image(prompt: str, width: int = 1024, height: int = 1024, steps: int = 50, guidance_scale: float = 3.5, seed: Optional[int] = None, save_filename: Optional[str] = None) -> dict:
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(FLUX_DEV_SNAPSHOT, torch_dtype=torch.float16, local_files_only=True)
    pipe.enable_sequential_cpu_offload()
    w, h = ((width + 31) // 32) * 32, ((height + 31) // 32) * 32
    gen = torch.Generator("cuda").manual_seed(seed) if seed else torch.Generator("cuda")
    result = await asyncio.to_thread(lambda: pipe(prompt=prompt, width=w, height=h, num_inference_steps=steps, guidance_scale=guidance_scale, generator=gen).images[0])
    fname = save_filename or f"gen_{int(time.time())}.png"
    out_path = OUTPUT_DIR / fname
    await asyncio.to_thread(result.save, str(out_path))
    return {"output_path": str(out_path), "image_b64": _img_to_b64(result)}

@mcp.tool(description="Return GPU and system memory info.")
async def get_gpu_status() -> dict:
    gpu_info = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info[f"gpu{i}"] = {"name": torch.cuda.get_device_name(i), "mem_total": torch.cuda.get_device_properties(i).total_memory // 1024**2, "mem_allocated": torch.cuda.memory_allocated(i) // 1024**2}
    return {"gpu": gpu_info, "server": "newbot.lan"}

@mcp.tool(description="Detect and mask objects in image.")
async def mask_object(image_b64: str, text: str) -> dict:
    mask_b64 = await asyncio.to_thread(_segment_object_sync, image_b64, text)
    fname = f"mask_{int(time.time())}.png"
    out_path = OUTPUT_DIR / fname
    await asyncio.to_thread(Image.open(io.BytesIO(base64.b64decode(mask_b64))).save, str(out_path))
    return {"output_path": str(out_path), "mask_b64": mask_b64}

@mcp.tool(description="Modify objects in image while preserving overall structure and resolution.")
async def smart_edit(image_b64: str, text: str = "", target_description: str = "", strength: float = 0.75, seed: Optional[int] = None) -> dict:
    gc.collect()
    torch.cuda.empty_cache()
    flux_pipe = await asyncio.to_thread(_get_flux_pipe)
    original_img = _b64_to_img(image_b64)
    resized_img = _resize_for_flux(original_img)
    mask_img = None
    if target_description:
        resized_b64 = _img_to_b64(resized_img)
        mask_b64 = await asyncio.to_thread(_segment_object_sync, resized_b64, target_description)
        mask_img = Image.open(io.BytesIO(base64.b64decode(mask_b64))).convert("L")
    if mask_img is None:
        mask_img = Image.new("L", resized_img.size, 255)
    gen = torch.Generator("cuda").manual_seed(seed) if seed else torch.Generator("cuda")
    result = await asyncio.to_thread(lambda: flux_pipe(prompt=text, image=resized_img, mask_image=mask_img, strength=strength, num_inference_steps=28, guidance_scale=3.5, generator=gen).images[0])
    result = result.resize(original_img.size, Image.LANCZOS)
    fname = f"edit_{int(time.time())}.png"
    out_path = OUTPUT_DIR / fname
    await asyncio.to_thread(result.save, str(out_path))
    return {"output_path": str(out_path), "image_b64": _img_to_b64(result)}

@mcp.tool(description="Fill masked region with new content.")
async def inpaint_image(image_b64: str, prompt: str = "", mask_image_b64: Optional[str] = None, strength: float = 0.85, seed: Optional[int] = None) -> dict:
    gc.collect()
    torch.cuda.empty_cache()
    flux_pipe = await asyncio.to_thread(_get_flux_pipe)
    original_img = _b64_to_img(image_b64)
    resized_img = _resize_for_flux(original_img)
    mask_img = None
    if mask_image_b64:
        mask_data = base64.b64decode(mask_image_b64.split(",", 1)[1] if mask_image_b64.startswith("data:") else mask_image_b64)
        mask_img = Image.open(io.BytesIO(mask_data)).convert("L").resize(resized_img.size, Image.LANCZOS)
    elif prompt:
        target_word = prompt.split()[0]
        resized_b64 = _img_to_b64(resized_img)
        mask_b64 = await asyncio.to_thread(_segment_object_sync, resized_b64, target_word)
        mask_img = Image.open(io.BytesIO(base64.b64decode(mask_b64))).convert("L")
    if mask_img is None: mask_img = Image.new("L", resized_img.size, 255)
    gen = torch.Generator("cuda").manual_seed(seed) if seed else torch.Generator("cuda")
    result = await asyncio.to_thread(lambda: flux_pipe(prompt=prompt, image=resized_img, mask_image=mask_img, strength=strength, num_inference_steps=28, guidance_scale=3.5, generator=gen).images[0])
    result = result.resize(original_img.size, Image.LANCZOS)
    fname = f"inpaint_{int(time.time())}.png"
    out_path = OUTPUT_DIR / fname
    await asyncio.to_thread(result.save, str(out_path))
    return {"output_path": str(out_path), "image_b64": _img_to_b64(result)}


async def _run_with_timeout():
    import uvicorn
    from mcp.server.fastmcp import FastMCP
    starlette_app = mcp.sse_app()
    config = uvicorn.Config(
        starlette_app,
        host=mcp.settings.host,
        port=mcp.settings.port,
        log_level=mcp.settings.log_level.lower(),
        timeout_keep_alive=1800,  # 30 minutes - image gen takes 5-10 min
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.settings.transport_security = None
    # Use custom run with increased keepalive timeout
    asyncio.run(_run_with_timeout())
