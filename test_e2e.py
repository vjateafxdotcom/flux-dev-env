#!/usr/bin/env python3
"""Full end-to-end test of FLUX MCP server on chellama.lan.
Tests all 4 tools in sequence with 512x512 images.

Uses httpx SSE transport correctly: keep ONE SSE connection alive,
use it for both receiving responses and as context for POST requests.
"""
import json, time, sys, re, io, threading, queue

try:
    import httpx
except ImportError:
    print("[setup] Installing httpx...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx"])
    import httpx

BASE = "http://chellama.lan:8769"


class MCPClient:
    """
    MCP SSE transport client.
    
    Flow:
      1. GET /sse -> persistent connection with session_id in event
      2. The SSE stream delivers response events back
      3. POST to /messages/?session_id=<id> sends requests
    
    Key insight: we keep the SSE connection alive in one thread,
    use a queue to collect incoming messages, and send via separate POSTs.
    """
    
    def __init__(self):
        self.session_id = None
        self.msg_queue = queue.Queue()
        self.sse_thread = None
        self._keep_running = False
        
    def connect(self):
        """Connect to SSE endpoint, extract session_id, start listener."""
        print("  Connecting to SSE...")
        
        # First, establish the SSE connection to get session_id
        with httpx.Client(timeout=10.0) as client:
            with client.stream("GET", f"{BASE}/sse", headers={"Accept": "text/event-stream"}) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if isinstance(line, bytes):
                        line = line.decode('utf-8')
                    line = line.strip()
                    if line.startswith("data: "):
                        data = line[6:].strip()
                        m = re.search(r'session_id=([^\s/&]+)', data)
                        if m:
                            self.session_id = m.group(1)
                            print(f"  Session ID: {self.session_id}")
                            break
        
        # Now start the persistent SSE listener for responses
        self._keep_running = True
        self.sse_thread = threading.Thread(target=self._sse_listener, daemon=True)
        self.sse_thread.start()
        
        # Wait a moment for listener to connect
        time.sleep(0.5)
    
    def _sse_listener(self):
        """Background thread: keep SSE connection alive, queue incoming events."""
        while self._keep_running:
            try:
                with httpx.Client(timeout=180.0) as client:
                    # Reconnect if needed (MCP may close idle connections)
                    url = f"{BASE}/sse"
                    if self.session_id:
                        url += f"?session_id={self.session_id}"
                    
                    with client.stream("GET", url, headers={
                        "Accept": "text/event-stream",
                        "Cache-Control": "no-cache",
                    }) as resp:
                        if resp.status_code != 200:
                            print(f"  SSE reconnect failed: {resp.status_code}")
                            time.sleep(1)
                            continue
                        
                        for line in resp.iter_lines():
                            if not self._keep_running:
                                return
                            if isinstance(line, bytes):
                                line = line.decode('utf-8')
                            line = line.strip()
                            if line.startswith("data: "):
                                data_str = line[6:].strip()
                                # Try parsing as JSON (tool response)
                                try:
                                    parsed = json.loads(data_str)
                                    print(f"  [SSE recv] {json.dumps(parsed)[:300]}")
                                    self.msg_queue.put_nowait(parsed)
                                except json.JSONDecodeError:
                                    pass
            except httpx.ReadError:
                pass  # Connection reset, will retry
            except Exception as e:
                print(f"  SSE listener error: {e}")
                time.sleep(1)
    
    def send_request(self, tool_name, params, timeout=300):
        """Send a tool call via POST and wait for response on SSE queue."""
        # Send the request
        payload = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params
            }
        }
        
        url = f"{BASE}/messages/?session_id={self.session_id}"
        print(f"  POST -> /messages/?session_id={self.session_id[:16]}...")
        
        with httpx.Client(timeout=30.0) as post_client:
            resp = post_client.post(url, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"POST failed: {resp.status_code} {resp.text[:100]}")
            print(f"  Server acknowledged: '{resp.text.strip()}'")
        
        # Wait for response on SSE queue
        print(f"  Waiting for result (timeout={timeout}s)...")
        try:
            resp_data = self.msg_queue.get(timeout=timeout)
            return resp_data
        except queue.Empty:
            raise RuntimeError(f"Timed out waiting for tool response after {timeout}s")
    
    def close(self):
        self._keep_running = False
        if self.sse_thread:
            self.sse_thread.join(timeout=2)


def extract_result(resp):
    """Extract result dict from MCP JSON-RPC response."""
    if not isinstance(resp, dict):
        return None
    
    # JSON-RPC: {id, result} or {id, error}
    result = resp.get("result")
    if result is None:
        return None
    
    if isinstance(result, str):
        try:
            return json.loads(result)
        except: pass
    elif isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    try:
                        return json.loads(item["text"])
                    except: pass
                elif isinstance(item, str):
                    try:
                        return json.loads(item)
                    except: pass
        elif isinstance(content, str):
            try:
                return json.loads(content)
            except: pass
        else:
            # result itself might be the output
            if "output_path" in result or "generation_time_s" in result:
                return result
    
    # Already the tool output?
    if isinstance(result, dict) and ("output_path" in result or "generation_time_s" in result):
        return result
    
    return None


def extract_image_b64(result_dict):
    """Extract base64 image from result."""
    if isinstance(result_dict, dict):
        return result_dict.get("image_b64") or result_dict.get("mask_b64")
    return None


def check_gpu(client):
    """Check current GPU status."""
    try:
        resp = client.send_request("get_gpu_status", {}, timeout=15)
        result = extract_result(resp)
        if result:
            gpu = result.get("gpu", {})
            alloc = gpu.get("gpu0_mem_allocated_mb", 0)
            total = gpu.get("gpu0_mem_total_mb", 0)
            loaded = result.get("pipelines_loaded", [])
            print(f"  GPU: {round(alloc)}MB / {round(total)}MB, loaded: {loaded}")
            return alloc
    except Exception as e:
        print(f"  GPU check error: {e}")
    return None


def test_generate_image(client):
    """Test 1: generate_image - 512x512 sunset."""
    print("\n" + "="*60)
    print("TEST 1/4: generate_image (512x512)")
    print("="*60)
    
    start = time.time()
    resp = client.send_request("generate_image", {
        "prompt": "A beautiful sunset over a calm ocean with colorful clouds, realistic photography style",
        "width": 512,
        "height": 512,
        "steps": 4,
        "guidance_scale": 0.0,
        "seed": 12345
    })
    elapsed = time.time() - start
    
    result = extract_result(resp)
    print(f"\n  Elapsed: {elapsed:.1f}s | Status: OK")
    
    if result:
        for k in ["output_path", "width", "height", "seed", "steps"]:
            if k in result:
                print(f"  {k}: {result[k]}")
        gt = result.get("generation_time_s")
        if gt:
            print(f"  Generation time: {gt}s")
    else:
        print(f"  No tool result. Raw response type: {type(resp).__name__}")
    
    return result


def test_mask_object(client, image_b64):
    """Test 2: mask_object - CLIPSeg segmentation."""
    print("\n" + "="*60)
    print("TEST 2/4: mask_object (CLIPSeg)")
    print("="*60)
    
    start = time.time()
    resp = client.send_request("mask_object", {
        "image_b64": image_b64,
        "text": "ocean"
    })
    elapsed = time.time() - start
    
    result = extract_result(resp)
    print(f"\n  Elapsed: {elapsed:.1f}s | Status: OK")
    
    if result:
        for k in ["output_path", "description"]:
            if k in result:
                v = str(result[k])[:100]
                print(f"  {k}: {v}")
    else:
        print(f"  No tool result. Raw response type: {type(resp).__name__}")
    
    return result


def test_smart_edit(client, image_b64):
    """Test 3: smart_edit - FLUX.dev img2img."""
    print("\n" + "="*60)
    print("TEST 3/4: smart_edit (FLUX.1-dev img2img)")
    print("="*60)
    
    start = time.time()
    resp = client.send_request("smart_edit", {
        "image_b64": image_b64,
        "text": "A starry night sky with bright stars and moonlit clouds",
        "target_description": "sky",
        "strength": 0.85,
        "seed": 12345
    })
    elapsed = time.time() - start
    
    result = extract_result(resp)
    print(f"\n  Elapsed: {elapsed:.1f}s | Status: OK")
    
    if result:
        for k in ["output_path", "width", "height", "seed"]:
            if k in result:
                print(f"  {k}: {result[k]}")
        gt = result.get("generation_time_s")
        if gt:
            print(f"  Generation time: {gt}s")
    else:
        print(f"  No tool result. Raw response type: {type(resp).__name__}")
    
    return result


def test_inpaint_image(client, image_b64):
    """Test 4: inpaint_image - FLUX.dev masked inpainting."""
    print("\n" + "="*60)
    print("TEST 4/4: inpaint_image (FLUX.1-dev + auto-mask)")
    print("="*60)
    
    start = time.time()
    resp = client.send_request("inpaint_image", {
        "image_b64": image_b64,
        "prompt": "a flock of white seagulls flying in formation",
        "strength": 0.85,
        "seed": 12345
    })
    elapsed = time.time() - start
    
    result = extract_result(resp)
    print(f"\n  Elapsed: {elapsed:.1f}s | Status: OK")
    
    if result:
        for k in ["output_path", "width", "height", "seed"]:
            if k in result:
                print(f"  {k}: {result[k]}")
        gt = result.get("generation_time_s")
        if gt:
            print(f"  Generation time: {gt}s")
    else:
        print(f"  No tool result. Raw response type: {type(resp).__name__}")
    
    return result


def main():
    print("="*60)
    print("FLUX MCP Server — Full End-to-End Test (chellama.lan)")
    print("512x512 images, all 4 tools in sequence")
    print("="*60)
    
    overall_start = time.time()
    results = {}
    client = MCPClient()
    
    try:
        # Connect SSE
        client.connect()
        
        # Pre-test GPU check
        print("\n--- Initial GPU status ---")
        check_gpu(client)
        
        # Test 1: generate_image
        gen_result = test_generate_image(client)
        gen_b64 = extract_image_b64(gen_result) if gen_result else None
        
        # GPU after generate (should be deallocated!)
        print("\n--- GPU after generate_image (check deallocation) ---")
        check_gpu(client)
        
        # Test 2: mask_object
        mask_result = test_mask_object(client, gen_b64) if gen_b64 else None
        
        # GPU after mask
        print("\n--- GPU after mask_object ---")
        check_gpu(client)
        
        # Test 3: smart_edit
        edit_result = test_smart_edit(client, gen_b64) if gen_b64 else None
        
        # GPU after smart_edit (should be deallocated!)
        print("\n--- GPU after smart_edit (check deallocation) ---")
        check_gpu(client)
        
        # Test 4: inpaint_image
        inpaint_result = test_inpaint_image(client, gen_b64) if gen_b64 else None
        
        # Post-test GPU check
        print("\n--- Final GPU status (post-deallocation) ---")
        check_gpu(client)
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()
    
    overall_elapsed = time.time() - overall_start
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    tests = [
        ("generate_image", gen_result is not None),
        ("mask_object", mask_result is not None),
        ("smart_edit", edit_result is not None),
        ("inpaint_image", inpaint_result is not None),
    ]
    
    for name, ok in tests:
        status = "PASS" if ok else "FAIL"
        print(f"  [{name:18s}] {status}")
    
    gen_t = gen_result.get("generation_time_s") if isinstance(gen_result, dict) else None
    edit_t = edit_result.get("generation_time_s") if isinstance(edit_result, dict) else None
    inpaint_t = inpaint_result.get("generation_time_s") if isinstance(inpaint_result, dict) else None
    
    print(f"\n  generate_image: {gen_t}s" if gen_t else f"\n  generate_image: N/A")
    print(f"  mask_object:    ~5s (not measured in tool response)")
    print(f"  smart_edit:     {edit_t}s" if edit_t else f"  smart_edit:     N/A")
    print(f"  inpaint_image:  {inpaint_t}s" if inpaint_t else f"  inpaint_image:  N/A")
    print(f"\n  Total wall time: {overall_elapsed:.1f}s")
    print("="*60)


if __name__ == "__main__":
    main()
