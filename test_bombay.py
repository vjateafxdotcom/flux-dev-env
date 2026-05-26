#!/usr/bin/env python3
"""Bombay edit v8 — inpaint to preserve motorcycle + simple LANCZOS upscale.

Key approach:
1. Resize portrait source to fit in 960px while keeping aspect ratio → ~720x960
2. Create generous protective mask around motorcycle (center zone, black=keep)
3. Inpaint with Bombay prompt → FLUX replaces only unpainted areas (motorcycle preserved)
4. FLUX outputs square ~1024x1024 — upscale to 1920x1200 using LANCZOS

Note: Due to FLUX outputting at fixed resolution, some aspect-ratio distortion is inevitable.
We minimize it by centering the portrait content in the landscape frame with dark padding.
"""
import sys, time, os, asyncio, base64, io, numpy as np
from PIL import Image

sys.path.insert(0, "/home/eafxadministrator/flux-dev-env")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from mcp_server import inpaint_image


def create_motorcycle_protection_mask(img_w, img_h):
    """Protect center zone (motorcycle/rider area). Black=keep, white=inpaint."""
    mask = np.full((img_h, img_w), 255, dtype=np.uint8)
    
    cx, cy = img_w // 2, img_h // 2
    
    # Very generous ellipse protecting ~60% width x ~70% height of center-lower area
    rx = int(img_w * 0.32)
    ry = int(img_h * 0.38)
    ellipse_y_center = cy + int(img_h * 0.15)
    
    y, x = np.ogrid[:img_h, :img_w]
    ellipse = ((x - cx)**2 / rx**2 + (y - ellipse_y_center)**2 / ry**2) <= 1.0
    
    mask[ellipse] = 0  # protected
    return Image.fromarray(mask, mode='L')


async def main():
    src_path = "/home/eafxadministrator/flux-dev-env/output/bombay_source.jpg"
    
    print("Loading source image...")
    src = Image.open(src_path).convert("RGB")
    orig_w, orig_h = src.size
    print(f"  Original: {orig_w}x{orig_h}")
    
    # Resize to fit in 960px max dimension (portrait maintains aspect ratio)
    max_dim = 960
    scale = min(max_dim / orig_w, max_dim / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    resized = src.resize((new_w, new_h), Image.LANCZOS)
    print(f"  Resized: {new_w}x{new_h}")
    
    # Create protective mask
    print("  Creating protective mask...")
    mask_img = create_motorcycle_protection_mask(new_w, new_h)
    mask_np = np.array(mask_img)
    protected_pct = (mask_np == 0).sum() / mask_np.size * 100
    print(f"  Protecting {protected_pct:.0f}% of image")
    
    mask_img.save("/home/eafxadministrator/flux-dev-env/output/mask_bombay_v8.png")
    
    # Convert to base64
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=95)
    image_b64 = base64.b64encode(buf.getvalue()).decode()
    
    mask_buf = io.BytesIO()
    mask_img.save(mask_buf, format="PNG")
    mask_b64 = base64.b64encode(mask_buf.getvalue()).decode()
    
    prompt = (
        "A busy Mumbai Bombay India street scene with heavy traffic. "
        "Auto-rickshaws, taxis, buses on a crowded road lined with Indian shops and buildings. "
        "Overhead power lines, dust, warm afternoon sunlight. Realistic photography."
    )
    
    # Step 1: Inpaint (motorcycle protected)
    print(f"\nRunning inpaint_image...")
    t0 = time.time()
    result = await inpaint_image(
        image_b64=image_b64,
        prompt=prompt,
        mask_image_b64=mask_b64,
        strength=0.85,
        seed=42
    )
    elapsed = time.time() - t0
    
    print(f"  Completed in {elapsed:.1f}s")
    
    if not isinstance(result, dict) or "image_b64" not in result:
        print(f"  ERROR: {result}")
        return None
    
    img_b64 = result["image_b64"]
    if img_b64.startswith("data:"):
        img_b64 = img_b64.split(",", 1)[1]
    
    img_data = base64.b64decode(img_b64)
    inpainted = Image.open(io.BytesIO(img_data)).convert('RGB')
    print(f"  FLUX output: {inpainted.size[0]}x{inpainted.size[1]}")
    
    # Step 2: Composite to 1920x1200 with dark padding (no stretching of content)
    target_w, target_h = 1920, 1200
    
    # Scale inpaint result to fit within target while maintaining aspect ratio
    scale_factor = min(target_w / inpainted.width, target_h / inpainted.height)
    new_w = int(inpainted.width * scale_factor)
    new_h = int(inpainted.height * scale_factor)
    
    scaled = inpainted.resize((new_w, new_h), Image.LANCZOS)
    
    # Center in 1920x1200 canvas with dark warm padding (Indian street vibe)
    canvas = Image.new('RGB', (target_w, target_h), (35, 28, 22))  # dark warm brown
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas.paste(scaled, (x_off, y_off))
    
    out_path = "/home/eafxadministrator/flux-dev-env/output/bombay_v8_final.png"
    canvas.save(out_path, "PNG")
    
    print(f"\n  Saved: {out_path}")
    print(f"  Final dimensions: {canvas.size[0]}x{canvas.size[1]} (exactly 1920x1200)")
    print(f"  File size: {os.path.getsize(out_path)} bytes")
    
    return result


if __name__ == "__main__":
    asyncio.run(main())
