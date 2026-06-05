#!/usr/bin/env python3
"""FLUX MCP Client - Run on chellama.lan to connect to chellama.lan:8769"""
import asyncio
import json
import base64
from pathlib import Path

from mcp.client.sse import sse_client
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCRequest, JSONRPCNotification, InitializeRequest


async def main():
    save_path = "/tmp/crocodile_cats_jungle.png"
    
    prompt = (
        "Four realistic cats eating the carcass of a crocodile in the jungle, "
        "hyper-realistic wildlife photography, detailed fur and scales, "
        "tropical jungle background with dappled sunlight, "
        "National Geographic style, 8K resolution, highly detailed, "
        "photo-realistic, cinematic lighting, dramatic scene"
    )
    
    print("=== FLUX Image Generation ===")
    print(f"Save to: {save_path}")
    print(f"Resolution: 1024x768")
    print(f"Prompt: {prompt[:80]}...")
    print()
    
    async with sse_client(
        "http://localhost:8769/sse",
        timeout=600.0,
        sse_read_timeout=600.0
    ) as (receive_stream, send_stream):
        # 1. Initialize
        print("1. Initializing...")
        await send_stream.send(SessionMessage(message=InitializeRequest(
            jsonrpc="2.0",
            id=1,
            method="initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "flux-gen-client",
                    "version": "1.0.0"
                }
            }
        )))
        init_resp = await receive_stream.receive()
        print(f"   Init OK: {init_resp}")
        
        # 2. Initialize notification
        print("2. Sending initialized notification...")
        await send_stream.send(SessionMessage(message=JSONRPCNotification(
            jsonrpc="2.0",
            method="notifications/initialized"
        )))
        
        # 3. Generate image
        print("3. Generating image... (this will take 2-5 minutes)")
        await send_stream.send(SessionMessage(message=JSONRPCRequest(
            jsonrpc="2.0",
            id=2,
            method="tools/call",
            params={
                "name": "generate_image",
                "arguments": {
                    "prompt": prompt,
                    "width": 1024,
                    "height": 768,
                    "save_filename": save_path
                }
            }
        )))
        
        # 4. Wait for result
        print("   Waiting for result...")
        result = await receive_stream.receive()
        print(f"\nResult: {result}")
        
        # Parse the SessionMessage
        msg = result.message
        print(f"\nMessage type: {type(msg)}")
        print(f"Message dump keys: {msg.model_dump().keys()}")
        
        # Get the root (JSONRPCResponse)
        root = msg.model_dump()["root"]
        print(f"\nRoot type: {type(root)}")
        print(f"Root keys: {root.keys()}")
        
        # Check for error
        if "error" in root:
            print(f"\nERROR: {root['error']}")
            return
        
        # Get content - each item has type and text
        content = root.get("result", {}).get("content", [])
        for item in content:
            if item.get("type") == "text":
                text = item["text"]
                print(f"\nContent type: text")
                print(f"Content length: {len(text)} chars")
                print(f"Content (first 300 chars): {text[:300]}")
                
                # This is a JSON string - parse it
                data = json.loads(text)
                print(f"\nParsed data keys: {data.keys()}")
                
                # Check for errors
                if "error" in data:
                    print(f"\nData error: {data['error']}")
                    return
                
                # Get the base64 image data
                if "image_b64" in data:
                    image_b64 = data["image_b64"]
                    print(f"\n✓ SUCCESS!")
                    print(f"   Output path: {data.get('output_path', 'N/A')}")
                    print(f"   Image B64 length: {len(image_b64)} chars")
                    print(f"   Width: {data.get('width', 'N/A')}")
                    print(f"   Height: {data.get('height', 'N/A')}")
                    
                    # Decode and save
                    print(f"\nDecoding base64 and saving...")
                    image_data = base64.b64decode(image_b64)
                    print(f"   Image size: {len(image_data):,} bytes ({len(image_data) // 1024} KB)")
                    
                    # Save to Desktop on chellama.lan
                    desktop_dir = Path("/home/eafxadministrator/Desktop")
                    desktop_dir.mkdir(parents=True, exist_ok=True)
                    desktop_file = desktop_dir / "crocodile_cats_jungle.png"
                    
                    with open(desktop_file, "wb") as f:
                        f.write(image_data)
                    
                    print(f"\n✓ Image saved to Desktop: {desktop_file}")
                    print(f"  File size: {desktop_file.stat().st_size:,} bytes")
                    
                    # Also save to /tmp for easy access
                    with open(save_path, "wb") as f:
                        f.write(image_data)
                    print(f"  Also saved to /tmp: {save_path}")
                    
                else:
                    print(f"\nResult (no image_b64): {json.dumps(data, indent=2)[:500]}")


if __name__ == "__main__":
    asyncio.run(main())
