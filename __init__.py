"""
comfyui-seedance-nodes
======================
ComfyUI custom nodes for BytePlus / ByteDance Seedance 2 video-generation API.

Install
-------
1.  Clone this repo into your  ComfyUI/custom_nodes/  directory.
2.  Install dependencies:
        pip install -r requirements.txt
3.  Set your API key as an environment variable:
        export ARK_API_KEY="your-api-key-here"
    or paste it directly in the api_key field of any node.
4.  Restart ComfyUI.

Nodes available after installation
-----------------------------------
  • Seedance – Video Generator          (unified node, auto-detects mode)
  • Seedance – Text to Video
  • Seedance – Image to Video (First Frame)
  • Seedance – Image to Video (First + Last Frame)
  • Seedance – Image to Video (Reference)
  • ByteDance Seedream 5                 (image generation)
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
