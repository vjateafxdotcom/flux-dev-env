# Agents Overview

This repository contains the implementation of the Image Generation MCP Server.

## Server Agent
The MCP server acts as a specialized tool provider, exposing GPU-accelerated image models (FLUX, Qwen, CLIPSeg) to other agents via the Model Context Protocol. It handles:
- Pipeline loading and CPU/GPU offloading.
- Image preprocessing (aspect ratio correction).
- Post-processing (LANCZOS upscaling).
