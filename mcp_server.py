#!/usr/bin/env python3
"""MCP Image Generation Server for newbot.lan (3x RTX PRO 4000 Blackwell)

Tools:
    generate_image — Text-to-image with Qwen-Image (high quality, ~100-180s)
    get_gpu_status  — GPU and system memory info
    mask_object     — Segment objects described in natural language from an image
    smart_edit      — Replace/modify objects in an existing image via FLUX.dev img2img
    inpaint_image   — Fill masked region with new content using FLUX.dev img2img
"""
import os, sys, time, base64, io, gc, asyncio, argparse, numpy as np, functools
from pathlib import Path
from typing import Optional

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("HF_TOKEN", "")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

import torch
from PIL import Image
from diffusers import QwenImagePipeline, FluxPipeline
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from mcp.server.fastmcp import FastMCP

MODEL_ID = "Qwen/Qwen-Image"
FLUX_DEV_SNAPSHOT = "/home/eafxadministrator/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
CLIPSEG_MODEL = "CIDAS/clipseg-rd64-refined"
OUTPUT_DIR = Path("/home/eafxadministrator/flux-dev-env/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_mcp = None
_pipe = None
_flux_pipe = None
_mask_model = None
_mask_processor = None

# Create the MCP server instance at module level so decorators can register tools
mcp = FastMCP("MCP Image Generation Server (newbot)")


def _get_mcp():
    return mcp


def _get_clipseg():
    """Load CLIPSeg model for object segmentation."""
    global _mask_model, _mask_processor
    if _mask_model is not None:
        return _mask_model, _mask_processor
    print("[mask] Loading CLIPSeg for object segmentation...")
    t0 = time.time()
    _mask_processor = CLIPSEGProcessor.from_pretrained(CLIPSEG_MODEL)
    _mask_model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_MODEL)
    if torch.cuda.is_available():
        _mask_model = _mask_model.to("cuda")
    print(f"[mask] Loaded in {time.time()-t0:.1f}s")
    return _mask_model, _mask_processor


def _get_flux_pipe():
    """Load FLUX.1-dev for editing tasks (img2img + inpaint)."""
    global _flux_pipe
    if _flux_pipe is not None:
        return _flux_pipe
    print("[flux] Loading FLUX.1-dev for editing tasks...")
    t0 = time.time()
    try:
        _flux_pipe = FluxPipeline.from_pretrained(
            FLUX_DEV_SNAPSHOT, torch_dtype=torch.float16, local_files_only=True
        )
        _flux_pipe.enable_sequential_cpu_offload()
    except Exception as e:
        print(f"[flux] Failed to load cached FLUX.dev ({e}), downloading fresh...")
        _flux_pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16
        )
    print(f"[flux] Loaded in {time.time()-t0:.1f}s")
    return _flux_pipe


def _get_pipe():
    """Load Qwen-Image for text-to-image generation."""
    global _pipe
    if _pipe is not None:
        return _pipe
    print("[qwen] Loading Qwen-Image (fp16)...")
    t0 = time.time()
    try:
        _pipe = QwenImagePipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16
        )
        if torch.cuda.is_available():
            _pipe.enable_sequential_cpu_offload()
    except Exception as e:
        print(f"[qwen] Failed with bfloat16/balanced: {e}")
        print("[qwen] Falling back to float16 on first GPU...")
        _pipe = QwenImagePipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16
        )
    print(f"[qwen] Loaded in {time.time()-t0:.1f}s")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            used = round(torch.cuda.memory_allocated(i) / 1024**2, 1)
            total = round(torch.cuda.get_device_properties(i).total_memory / 1024**2, 1)
            print(f"[qwen] GPU{i}: {used}/{total} MB")
    return _pipe


def _img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _b64_to_img(b64_str):
    """Decode base64 image string (with or without data: URI prefix) to PIL Image."""
    if isinstance(b64_str, str) and b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _segment_object_sync(image_b64, text):
    """Synchronous function for segmenting an object — runs inside asyncio.to_thread()."""
    pipe_mask, proc = _get_clipseg()
    image = _b64_to_img(image_b64)

    inputs = proc(texts=[text], images=[image], return_tensors="pt")
    for k, v in inputs.items():
        if hasattr(v, "to"):
            inputs[k] = v.to(pipe_mask.device)

    with torch.no_grad():
        outputs = pipe_mask(**inputs)

    # Get mask from logits
    mask_np = torch.sigmoid(outputs.logits[0]).cpu().numpy()

    # Threshold using Otsu if available, else 0.5
    try:
        from skimage import measure as skm_measure
        thresh = skm_measure.threshold_otsu(mask_np)
        mask_bool = mask_np > thresh
    except ImportError:
        mask_bool = mask_np > 0.5

    # Morphological cleanup for clean edges
    from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure
    structure = generate_binary_structure(2, 1)
    mask_bool = binary_opening(mask_bool, structure=structure)
    mask_bool = binary_closing(mask_bool, structure=structure)

    # Scale to original image size if needed
    mask_img = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode='L')
    if mask_img.size != image.size:
        mask_img = mask_img.resize(image.size, Image.LANCZOS)

    return _img_to_b64(mask_img)


@functools.lru_cache(maxsize=1)
def _get_schnell_pipe():
    """Load FLUX.1-schnell for fast text-to-image generation."""
    snapshot = "/home/eafxadministrator/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-schnell/snapshots/741f7c3ce8b383c54771c7003378a50191e9efe9"
    print("[schnell] Loading FLUX.1-schnell for generation...")
    t0 = time.time()
    try:
        pipe = FluxPipeline.from_pretrained(
            snapshot, torch_dtype=torch.float16, local_files_only=True
        )
        pipe.enable_sequential_cpu_offload()
    except Exception as e:
        print(f"[schnell] Failed to load cached FLUX.schnell ({e}), downloading...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16
        )
    print(f"[schnell] Loaded in {time.time()-t0:.1f}s")
    return pipe


@mcp.tool(description="Generate an image from a text prompt using Qwen-Image on chellama.lan (RTX 4000 Ada, 20GB). Returns base64-encoded PNG and output file path.")
async def generate_image(prompt: str, width: int = 1024, height: int = 1024,
                         steps: int = 50, guidance_scale: float = 5.0,
                         seed: Optional[int] = None,
                         save_filename: Optional[str] = None) -> dict:
    """Generate with retry using FLUX.schnell (fast, reliable)."""
    # Force width/height to be multiples of 32 for FLUX.schnell
    w = ((width + 31) // 32) * 32
    h = ((height + 31) // 32) * 32
    
    pipe = _get_schnell_pipe()
    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    for attempt in range(2):
        try:
            t0 = time.time()
            result = await asyncio.to_thread(
                lambda: pipe(
                    prompt=prompt,
                    width=w,
                    height=h,
                    num_inference_steps=4,  # FLUX.schnell needs only 4 steps
                    guidance_scale=0.0,     #schnell doesn't use guidance_scale
                    generator=generator,
                ).images[0])
            gen_time = time.time() - t0
            fname = save_filename or f"qwen_{int(time.time())}_{seed}.png"
            out_path = Path(fname) if Path(fname).is_absolute() else OUTPUT_DIR / fname
            await asyncio.to_thread(result.save, str(out_path))
            return {"output_path": str(out_path), "image_b64": _img_to_b64(result),
                    "width": result.width, "height": result.height, "seed": seed,
                    "generation_time_s": round(gen_time, 1), "steps": 4,
                    "device": "newbot.lan (FLUX.1-schnell bfloat16)"}
        except Exception as e:
            err_msg = str(e)
            if "cudaError" in err_msg or "CUDADriver" in err_msg or "Triton" in err_msg:
                print(f"[generate_image] Attempt {attempt+1}/2 CUDA error: {err_msg[:200]}")
                gc.collect()
                torch.cuda.empty_cache()
                if attempt == 0:
                    continue
            raise e


@mcp.tool(description="Return GPU VRAM allocation, total/available RAM, and model pipeline load status.")
async def get_gpu_status() -> dict:
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info["cuda_available"] = True
        gpu_info["device_count"] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            gpu_info[f"gpu{i}_name"] = torch.cuda.get_device_name(i)
            gpu_info[f"gpu{i}_mem_total_mb"] = round(torch.cuda.get_device_properties(i).total_memory/1024**2, 1)
            gpu_info[f"gpu{i}_mem_allocated_mb"] = round(torch.cuda.memory_allocated(i)/1024**2, 1)
    mem = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line: mem["total_kb"] = int(line.split()[1])
                if "MemAvailable" in line: mem["available_kb"] = int(line.split()[1])
    except: pass
    ram = {}
    if mem:
        t = mem["total_kb"]//1024; a = mem.get("available_kb",0)//1024
        ram = {"total_mb": t, "available_mb": a, "used_mb": t-a}
    return {"gpu": gpu_info, "ram": ram, "pipeline_loaded": _pipe is not None,
            "model": "Qwen-Image + FLUX.1-dev (editing)", "server": "newbot.lan"}


@mcp.tool(description="Detect and mask objects described in natural language from an image. Returns a binary mask PNG (white=picked object, black=background). Use this before smart_edit or inpaint_image to isolate the target area.")
async def mask_object(image_b64: str, text: str) -> dict:
    """Segment an object from an image using CLIPSeg."""
    gc.collect()
    torch.cuda.empty_cache()

    mask_b64 = await asyncio.to_thread(_segment_object_sync, image_b64, text)

    fname = f"mask_{int(time.time())}.png"
    out_path = OUTPUT_DIR / fname
    mask_data = base64.b64decode(mask_b64)
    await asyncio.to_thread(Image.open(io.BytesIO(mask_data)).save, str(out_path))

    return {"output_path": str(out_path), "mask_b64": mask_b64,
            "description": f"Mask for '{text}' from provided image",
            "device": "newbot.lan (CLIPSeg rd64-refined, GPU)"}


@mcp.tool(description="Replace or modify objects in an existing image. Uses CLIPSeg to automatically target the region if target_description is provided, then FLUX.dev img2img for high-quality editing. Returns the edited image as base64 PNG.")
async def smart_edit(image_b64: str, text: str = "",
                     target_description: str = "",
                     strength: float = 0.75,
                     seed: Optional[int] = None) -> dict:
    """Edit an image using FLUX.dev img2img with optional CLIPSeg targeting."""
    gc.collect()
    torch.cuda.empty_cache()

    flux_pipe = await asyncio.to_thread(_get_flux_pipe)
    original_img = _b64_to_img(image_b64)

    # Create mask from target description if provided
    mask_img = None
    if target_description:
        print(f"[edit] Segmenting '{target_description}' from image...")
        mask_b64 = await asyncio.to_thread(_segment_object_sync, image_b64, target_description)
        mask_data = base64.b64decode(mask_b64)
        mask_img = Image.open(io.BytesIO(mask_data)).convert("L")

    strength = max(0.1, min(1.0, float(strength)))
    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()

    generator = torch.Generator(device="cuda").manual_seed(seed)
    t0 = time.time()

    # FLUX.dev supports image + mask_image parameters for img2img editing
    if mask_img is not None:
        result = await asyncio.to_thread(
            lambda: flux_pipe(
                prompt=text if text else "edit this image",
                image=original_img,
                mask_image=mask_img,
                strength=strength,
                num_inference_steps=28,
                guidance_scale=3.5,
                generator=generator,
            ).images[0])
    else:
        # Plain img2img without mask — whole image is editable
        result = await asyncio.to_thread(
            lambda: flux_pipe(
                prompt=text if text else "edit this image",
                image=original_img,
                strength=strength,
                num_inference_steps=28,
                guidance_scale=3.5,
                generator=generator,
            ).images[0])

    gen_time = time.time() - t0

    fname = f"edit_{int(time.time())}_{seed}.png"
    out_path = OUTPUT_DIR / fname
    await asyncio.to_thread(result.save, str(out_path))

    return {"output_path": str(out_path), "image_b64": _img_to_b64(result),
            "width": result.width, "height": result.height, "seed": seed,
            "generation_time_s": round(gen_time, 1), "strength": strength,
            "device": "newbot.lan (FLUX.1-dev img2img + CLIPSeg masking)",
            "mask_applied": mask_img is not None}


@mcp.tool(description="Fill a masked region of an image with new content described by prompt text. The mask defines which pixels to replace. If no mask is provided, the first word of the prompt is used to auto-segment via CLIPSeg.")
async def inpaint_image(image_b64: str, prompt: str = "",
                        mask_image_b64: Optional[str] = None,
                        strength: float = 0.85,
                        seed: Optional[int] = None) -> dict:
    """Inpaint masked regions using FLUX.dev img2img with mask conditioning."""
    gc.collect()
    torch.cuda.empty_cache()

    flux_pipe = await asyncio.to_thread(_get_flux_pipe)
    original_img = _b64_to_img(image_b64)

    # Process mask — use provided mask or auto-segment from prompt
    mask_img = None
    if mask_image_b64:
        mask_data = base64.b64decode(
            mask_image_b64.split(",", 1)[1] if mask_image_b64.startswith("data:") else mask_image_b64)
        mask_img = Image.open(io.BytesIO(mask_data)).convert("L")
        if mask_img.size != original_img.size:
            mask_img = mask_img.resize(original_img.size, Image.LANCZOS)
    elif prompt and not mask_image_b64:
        # Auto-segment using first meaningful word from prompt
        target_word = prompt.split()[0] if prompt else "object"
        print(f"[inpaint] Auto-segmenting '{target_word}' for inpainting...")
        mask_b64 = await asyncio.to_thread(_segment_object_sync, image_b64, target_word)
        mask_data = base64.b64decode(mask_b64)
        mask_img = Image.open(io.BytesIO(mask_data)).convert("L")

    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()

    generator = torch.Generator(device="cuda").manual_seed(seed)
    t0 = time.time()

    if mask_img is not None:
        result = await asyncio.to_thread(
            lambda: flux_pipe(
                prompt=prompt if prompt else "fill this area",
                image=original_img,
                mask_image=mask_img,
                strength=strength,
                num_inference_steps=28,
                guidance_scale=3.5,
                generator=generator,
            ).images[0])
    else:
        result = await asyncio.to_thread(
            lambda: flux_pipe(
                prompt=prompt if prompt else "fill this area",
                image=original_img,
                strength=strength,
                num_inference_steps=28,
                guidance_scale=3.5,
                generator=generator,
            ).images[0])

    gen_time = time.time() - t0

    fname = f"inpaint_{int(time.time())}_{seed}.png"
    out_path = OUTPUT_DIR / fname
    await asyncio.to_thread(result.save, str(out_path))

    return {"output_path": str(out_path), "image_b64": _img_to_b64(result),
            "width": result.width, "height": result.height, "seed": seed,
            "generation_time_s": round(gen_time, 1), "strength": strength,
            "device": "newbot.lan (FLUX.1-dev inpaint + CLIPSeg auto-masking)",
            "mask_used": mask_img is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    print(f"[mcp] MCP Image Generation Server on http://{args.host}:{args.port}/sse")
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.settings.transport_security = None
    mcp.run(transport="sse")
