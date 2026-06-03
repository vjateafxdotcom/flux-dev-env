# Image Generation MCP Server (FLUX + Qwen)

This server provides high-quality image generation, segmentation, and editing capabilities using the FLUX.1-dev and Qwen-Image models, hosted on `chellama.lan`.

## Core Tools
- `generate_image`: Text-to-image generation (FLUX.1-schnell).
- `get_gpu_status`: Monitors VRAM and system memory on the RTX 4000 Blackwell GPUs.
- `mask_object`: Natural language segmentation using CLIPSeg to create binary masks.
- `smart_edit`: High-level image modification that can automatically segment objects and apply img2img transformations.
- `inpaint_image`: Targeted region replacement using a mask and prompt.

## Integration with codeSync
The server is utilized across different agentic workflows:

### `codeSync/smarked/`
Used for visual asset generation and creative modification of images to support marketing and design tasks.

### `codeSync/research/`
Used for creating visual aids, diagrams, and conceptual representations of research findings to improve documentation and synthesis.

## Case Study: The Everglades Transformation
In a recent session, the server was used to transform a personal photo of a man and his dog into a realistic scene in the Florida Everglades.

**Workflow used:**
1. **Subject Preservation**: Used `mask_object` to identify the "person and dog".
2. **Background Targeting**: Inverted the subject mask to create a background-only target.
3. **High-Realism Inpainting**: Employed `inpaint_image` with a prompt emphasizing "National Geographic photography," "cypress knees," and "muted earthy tones."
4. **Resolution Management**: Leveraged the enhanced codebase to maintain the original photo's aspect ratio and upscale the final result back to the original resolution.

## Technical Specifications
- **Backend**: FLUX.1-dev (Schannel/Dev), CLIPSeg, Qwen-Image.
- **Hardware**: 3x RTX PRO 4000 Blackwell.
- **Transport**: SSE (Server-Sent Events) via FastMCP.
