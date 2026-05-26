#!/usr/bin/env python3
"""Direct async test of FLUX MCP tools on chellama.lan.
Single event loop, imports mcp_server directly, no HTTP/SSE.
"""
import sys, time, os, asyncio, base64, io, gc

sys.path.insert(0, "/home/eafxadministrator/flux-dev-env")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from mcp_server import (
    generate_image, mask_object, smart_edit, inpaint_image, get_gpu_status
)


async def check_gpu():
    result = await get_gpu_status()
    gpu = result.get("gpu", {})
    alloc = gpu.get("gpu0_mem_allocated_mb", 0)
    total = gpu.get("gpu0_mem_total_mb", 0)
    loaded = result.get("pipelines_loaded", [])
    print(f"  GPU: {round(alloc)}MB / {round(total)}MB, loaded: {loaded}")
    return round(alloc), round(total), loaded


def save_b64(b64_str, prefix):
    """Save base64 image to disk and return clean b64 (no data: URI)."""
    if not b64_str or isinstance(b64_str, str) and b64_str.startswith("data:"):
        if isinstance(b64_str, str) and b64_str.startswith("data:"):
            b64_str = b64_str.split(",", 1)[1]
        else:
            return None
    data = base64.b64decode(b64_str)
    from PIL import Image
    img = Image.open(io.BytesIO(data))
    path = f"/home/eafxadministrator/flux-dev-env/output/{prefix}.png"
    img.save(path)
    print(f"  Saved: {path} ({os.path.getsize(path)} bytes)")
    return b64_str


async def main():
    print("="*60)
    print("FLUX MCP Server — Direct Tool Test (chellama.lan)")
    print("512x512 images, all 4 tools in sequence")
    print("="*60)
    
    overall_start = time.time()
    gen_b64 = None
    
    # --- Initial GPU ---
    print("\n--- Initial GPU status ---")
    await check_gpu()
    
    # ---- TEST 1: generate_image ----
    print("\n" + "="*60)
    print("TEST 1/4: generate_image (512x512)")
    print("="*60)
    
    t0 = time.time()
    gen_result = await generate_image(
        prompt="A beautiful sunset over a calm ocean with colorful clouds, realistic photography style",
        width=512, height=512, steps=4, guidance_scale=0.0, seed=12345
    )
    elapsed = time.time() - t0
    
    print(f"\n  Total tool call: {elapsed:.1f}s")
    if isinstance(gen_result, dict):
        for k in ["output_path", "width", "height", "seed"]:
            print(f"  {k}: {gen_result[k]}")
        gt = gen_result.get("generation_time_s")
        if gt:
            print(f"  Generation time (GPU only): {gt}s")
        # Save image for next tools
        b64_raw = gen_result.get("image_b64", "")
        gen_b64 = save_b64(b64_raw, "gen_512_direct")
    
    # GPU after generate
    print("\n--- GPU after generate_image ---")
    await check_gpu()
    
    # ---- TEST 2: mask_object ----
    print("\n" + "="*60)
    print("TEST 2/4: mask_object (CLIPSeg)")
    print("="*60)
    
    if gen_b64 is None:
        print("  SKIP — no generated image")
    else:
        t0 = time.time()
        mask_result = await mask_object(image_b64=gen_b64, text="ocean")
        elapsed = time.time() - t0
        
        print(f"\n  Total tool call: {elapsed:.1f}s")
        if isinstance(mask_result, dict):
            for k in ["output_path", "description"]:
                print(f"  {k}: {str(mask_result[k])[:120]}")
    
    # GPU after mask
    print("\n--- GPU after mask_object ---")
    await check_gpu()
    
    # ---- TEST 3: smart_edit ----
    print("\n" + "="*60)
    print("TEST 3/4: smart_edit (FLUX.1-dev img2img)")
    print("="*60)
    
    if gen_b64 is None:
        print("  SKIP — no generated image")
    else:
        t0 = time.time()
        edit_result = await smart_edit(
            image_b64=gen_b64,
            text="A starry night sky with bright stars and moonlit clouds",
            target_description="sky", strength=0.85, seed=12345
        )
        elapsed = time.time() - t0
        
        print(f"\n  Total tool call: {elapsed:.1f}s")
        if isinstance(edit_result, dict):
            for k in ["output_path", "width", "height", "seed"]:
                print(f"  {k}: {edit_result[k]}")
            gt = edit_result.get("generation_time_s")
            if gt:
                print(f"  Generation time (GPU only): {gt}s")
    
    # GPU after smart_edit
    print("\n--- GPU after smart_edit ---")
    await check_gpu()
    
    # ---- TEST 4: inpaint_image ----
    print("\n" + "="*60)
    print("TEST 4/4: inpaint_image (FLUX.1-dev + auto-mask)")
    print("="*60)
    
    if gen_b64 is None:
        print("  SKIP — no generated image")
    else:
        t0 = time.time()
        inpaint_result = await inpaint_image(
            image_b64=gen_b64,
            prompt="a flock of white seagulls flying in formation",
            strength=0.85, seed=12345
        )
        elapsed = time.time() - t0
        
        print(f"\n  Total tool call: {elapsed:.1f}s")
        if isinstance(inpaint_result, dict):
            for k in ["output_path", "width", "height", "seed"]:
                print(f"  {k}: {inpaint_result[k]}")
            gt = inpaint_result.get("generation_time_s")
            if gt:
                print(f"  Generation time (GPU only): {gt}s")
    
    # Final GPU
    print("\n--- Final GPU status ---")
    await check_gpu()
    
    # Summary
    total_elapsed = time.time() - overall_start
    
    gen_t = gen_result.get("generation_time_s") if isinstance(gen_result, dict) else None
    edit_t = edit_result.get("generation_time_s") if isinstance(edit_result, dict) else None
    inpaint_t = inpaint_result.get("generation_time_s") if isinstance(inpaint_result, dict) else None
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, ok in [
        ("generate_image", gen_result is not None and isinstance(gen_result, dict)),
        ("mask_object", 'mask_result' in dir() and mask_result is not None and isinstance(mask_result, dict)),
        ("smart_edit", edit_result is not None and isinstance(edit_result, dict)),
        ("inpaint_image", inpaint_result is not None and isinstance(inpaint_result, dict)),
    ]:
        print(f"  [{name:18s}] {'PASS' if ok else 'FAIL'}")
    
    print(f"\n  GPU-only timing:")
    print(f"    generate_image: {gen_t}s" if gen_t else f"    generate_image: N/A")
    print(f"    mask_object:    ~5s (tool doesn't report)")
    print(f"    smart_edit:     {edit_t}s" if edit_t else f"    smart_edit:     N/A")
    print(f"    inpaint_image:  {inpaint_t}s" if inpaint_t else f"    inpaint_image:  N/A")
    print(f"\n  Total wall time: {total_elapsed:.1f}s")
    
    alloc, total, loaded = await check_gpu()
    print(f"\n  VRAM after all tests (should be freed): ~{alloc}MB / {total}MB")
    print(f"  Pipelines still loaded: {loaded}")
    if len(loaded) == 0 or loaded == []:
        print("  >>> Auto-deallocation WORKING - no pipelines remain!")
    else:
        print("  >>> WARNING: Some pipelines still loaded in VRAM")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
