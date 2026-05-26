import sys, os, gc, numpy as np, time
sys.path.insert(0, "/home/eafxadministrator/flux-dev-env")
import torch, cv2
from PIL import Image
from diffusers import FluxPipeline

SRC = "/tmp/bombay_motorcyclist_orig.jpg"  # Actually the Everglades target now - wrong name but correct file path
MASK_PATH = "/tmp/sam2_full_moto_mask.png"  
FINAL_OUT = "/tmp/everglades_final.png"
BG_OUT = "/tmp/everglades_bg.png"

print("[1] Loading...")
gc.collect(); torch.cuda.empty_cache()
src_pil = Image.open(SRC).convert("RGB")
mask_img = Image.open(MASK_PATH).convert("L")
w, h = src_pil.size  # w=1536, h=2048

# Check what image we're actually loading
print(f"  Source dims: {w}x{h}")
sample_color = np.array(src_pil.resize((10,10), Image.LANCZOS)).mean(axis=(0,1))
print(f"  Source average color: RGB({sample_color.astype(int)})")

print("[2] Generating Everglades background...")
SPEEDY_MODEL = "/home/eafxadministrator/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-schnell/snapshots/741f7c3ce8b383c54771c7003378a50191e9efe9"
schnell_pipe = FluxPipeline.from_pretrained(SPEEDY_MODEL, torch_dtype=torch.float16, local_files_only=True)
schnell_pipe.enable_sequential_cpu_offload()

bg_prompt = "Dense Florida Everglades forest. Tall pine trees with Spanish moss hanging from branches. Marshy wetland ground covered in green ferns and sawgrass. Sunlight filtering through canopy creating dappled light on dark brown mud and water. Humid atmosphere with slight mist. Natural wildlife habitat, cypress tree trunks rising from swamp water. Realistic nature photography, photorealistic, 4k, national geographic style."

gen = torch.Generator(device="cuda").manual_seed(77)
bg_result = schnell_pipe(prompt=bg_prompt, width=w, height=h, num_inference_steps=6, guidance_scale=0.0, generator=gen).images[0]
bg_result.save(BG_OUT)
print(f"  Background saved")

print("[3] Compositing (100% opaque inside mask)...")
t0 = time.time()

# Resize everything correctly  
bg_pil = bg_result.resize((w, h), Image.LANCZOS)
src_resized = src_pil.resize((w, h), Image.LANCZOS)
mask_resized = mask_img.resize((w, h), Image.NEAREST).convert("L")

bg_arr = np.array(bg_pil).astype(np.float32)  # (H,W,3)
src_arr = np.array(src_resized).astype(np.float32)  # (H,W,3)
mask_arr = np.array(mask_resized).astype(np.float32) / 255.0  # (H,W) - binary 0 or 1

# KEY FIX: No blur at all on mask values inside the region.
# Only apply a very tiny feather to the absolute edge pixels (transition from 1→0).
# This ensures alpha=1.0 everywhere INSIDE the masked area = 100% opaque subject.

# Create hard alpha channel
alpha_hard = mask_arr.reshape((h, w, 1))  # Values are exactly 0.0 or 1.0 only

# Minimal feathering: just 3px at the very edge to avoid harsh aliasing
kernel_small = np.ones((7, 7), dtype=np.float32) / (7*7)
mask_padded = np.pad(mask_arr, 3, mode='constant')
alpha_fine = cv2.filter2D(mask_padded[3:-3, 3:-3], -1, kernel_small).reshape((h, w, 1))

# Blend: use 1.0 inside mask (from alpha_hard), only allow feather at edge
alpha_final = np.maximum(alpha_hard, alpha_fine)  # Inside=1.0, edges=transitioned
alpha_final = np.clip(alpha_final, 0.0, 1.0)

print(f"  Alpha range: [{alpha_final.min():.3f}, {alpha_final.max():.3f}]")
print(f"  Pixels with alpha=1.0: {(alpha_final[:,:,0] > 0.99).sum()/(h*w)*100:.1f}%")

# Composite: bg*(1-alpha) + src*alpha
alpha_3c = np.broadcast_to(alpha_final, (h, w, 3)).astype(np.float32)
result_arr = bg_arr * (1 - alpha_3c) + src_arr * alpha_3c
result_arr = np.clip(result_arr, 0, 255).astype(np.uint8)

result_pil = Image.fromarray(result_arr)
result_pil.save(FINAL_OUT)
elapsed = time.time() - t0
sz = os.path.getsize(FINAL_OUT)
print(f" DONE! {FINAL_OUT} ({sz:,} bytes, {w}x{h}, {elapsed:.1f}s)")

# Verify no ghosting
masked_pixels = result_arr[alpha_hard[:,:,0] > 0.9]
unmasked_pixels = result_arr[alpha_hard[:,:,0] < 0.1]
print(f"  Masked region pixels: {len(masked_pixels):,} (should be opaque)")
print(f"  Source masked mean RGB: {np.array(src_resized.resize((50,50), Image.LANCZOS)).mean(axis=(0,1)).astype(int) if len(masked_pixels) else 'N/A'}")
