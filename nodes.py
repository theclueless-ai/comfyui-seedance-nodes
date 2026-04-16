"""
comfyui-seedance-nodes  –  nodes.py
====================================
ComfyUI custom nodes for BytePlus / ByteDance Seedance 2 video-generation API.

Nodes
-----
SeedanceTextToVideo          – Text → Video
SeedanceI2VFirstFrame        – Image-to-Video  (first frame)
SeedanceI2VFirstLastFrame    – Image-to-Video  (first + last frame)
SeedanceI2VReference         – Image-to-Video  (reference image style)
"""

import os

try:
    import folder_paths
    OUTPUT_DIR = folder_paths.get_output_directory()
except ImportError:
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

from .api_client import SeedanceAPIClient
from .utils import tensor_to_base64, download_video, resolve_api_key


# ──────────────────────────────────────────────────────────────────
# Shared constants
# ──────────────────────────────────────────────────────────────────

SEEDANCE_MODELS = [
    "dreamina-seedance-2-0-260128",
]

# The reference-image I2V mode uses a different, dedicated model
SEEDANCE_REFERENCE_MODELS = [
    "seedance-1-0-lite-i2v-250428",
    "dreamina-seedance-2-0-260128",  # kept as fallback option
]

RATIO_OPTIONS  = ["16:9", "9:16", "1:1", "4:3", "3:4", "adaptive"]
DURATION_OPTIONS = [5, 10]   # seconds – values accepted by the API


# ──────────────────────────────────────────────────────────────────
# Helper: build client + run task
# ──────────────────────────────────────────────────────────────────

def _run_task(api_key: str, payload: dict, poll_interval: int, max_wait: int):
    """Create a task, poll until done, download the video, return (url, path)."""
    client = SeedanceAPIClient(api_key)
    task_id = client.create_task(payload)
    print(f"[Seedance] Task created: {task_id}")
    result  = client.poll_task(task_id, poll_interval=poll_interval, max_wait=max_wait)

    video_url = result.get("content", {}).get("video_url", "")
    if not video_url:
        raise RuntimeError(f"[Seedance] No video_url in response: {result}")

    video_path = download_video(video_url, OUTPUT_DIR, prefix="seedance")
    return video_url, video_path


# ──────────────────────────────────────────────────────────────────
# Node 1 – Text to Video
# ──────────────────────────────────────────────────────────────────

class SeedanceTextToVideo:
    """
    Generate a video from a text prompt using the Seedance 2 API (T2V mode).
    """

    CATEGORY    = "Seedance/Video Generation"
    FUNCTION    = "generate"
    RETURN_TYPES  = ("STRING", "STRING")
    RETURN_NAMES  = ("video_url", "video_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "BytePlus ARK API key. Leave empty to use ARK_API_KEY env variable.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A girl holding a fox, the camera slowly pulls out.",
                    "tooltip": "Text description of the video to generate.",
                }),
                "model": (SEEDANCE_MODELS, {"default": SEEDANCE_MODELS[0]}),
                "ratio": (RATIO_OPTIONS, {"default": "16:9"}),
                "duration": (DURATION_OPTIONS, {"default": 5}),
                "generate_audio": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": False}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60, "step": 5,
                                          "tooltip": "Seconds between status-check requests."}),
                "max_wait": ("INT",  {"default": 600, "min": 60, "max": 3600, "step": 60,
                                      "tooltip": "Maximum seconds to wait for the task."}),
            }
        }

    def generate(
        self,
        api_key,
        prompt,
        model,
        ratio,
        duration,
        generate_audio,
        watermark,
        poll_interval,
        max_wait,
    ):
        key = resolve_api_key(api_key)
        payload = {
            "model": model,
            "content": [{"type": "text", "text": prompt}],
            "ratio": ratio,
            "duration": duration,
            "generate_audio": generate_audio,
            "watermark": watermark,
        }
        video_url, video_path = _run_task(key, payload, poll_interval, max_wait)
        return (video_url, video_path)


# ──────────────────────────────────────────────────────────────────
# Node 2 – Image-to-Video  (first frame)
# ──────────────────────────────────────────────────────────────────

class SeedanceI2VFirstFrame:
    """
    Generate a video from a first-frame image + text prompt (I2V mode).

    The supplied image is used as the first frame; the model animates it
    forward according to the text description.
    """

    CATEGORY    = "Seedance/Video Generation"
    FUNCTION    = "generate"
    RETURN_TYPES  = ("STRING", "STRING")
    RETURN_NAMES  = ("video_url", "video_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "first_frame": ("IMAGE", {
                    "tooltip": "ComfyUI IMAGE tensor used as the first frame of the video.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "The camera slowly zooms out.",
                    "tooltip": "Describes how the image should be animated.",
                }),
                "model": (SEEDANCE_MODELS, {"default": SEEDANCE_MODELS[0]}),
                "ratio": (RATIO_OPTIONS, {"default": "adaptive"}),
                "duration": (DURATION_OPTIONS, {"default": 5}),
                "generate_audio": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": False}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60, "step": 5}),
                "max_wait":      ("INT",  {"default": 600, "min": 60, "max": 3600, "step": 60}),
            },
            "optional": {
                "image_url_override": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional: paste a direct HTTP URL to use instead of the IMAGE input.",
                }),
            },
        }

    def generate(
        self,
        api_key,
        first_frame,
        prompt,
        model,
        ratio,
        duration,
        generate_audio,
        watermark,
        poll_interval,
        max_wait,
        image_url_override="",
    ):
        key = resolve_api_key(api_key)

        if image_url_override and image_url_override.strip().startswith("http"):
            img_url = image_url_override.strip()
        else:
            img_url = tensor_to_base64(first_frame)

        payload = {
            "model": model,
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": img_url}},
            ],
            "ratio": ratio,
            "duration": duration,
            "generate_audio": generate_audio,
            "watermark": watermark,
        }
        video_url, video_path = _run_task(key, payload, poll_interval, max_wait)
        return (video_url, video_path)


# ──────────────────────────────────────────────────────────────────
# Node 3 – Image-to-Video  (first + last frame)
# ──────────────────────────────────────────────────────────────────

class SeedanceI2VFirstLastFrame:
    """
    Generate a video that starts at *first_frame* and ends at *last_frame*.

    Both images are sent with their respective roles so the model interpolates
    the motion between them guided by the text prompt.
    """

    CATEGORY    = "Seedance/Video Generation"
    FUNCTION    = "generate"
    RETURN_TYPES  = ("STRING", "STRING")
    RETURN_NAMES  = ("video_url", "video_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "first_frame": ("IMAGE", {"tooltip": "First frame of the output video."}),
                "last_frame":  ("IMAGE", {"tooltip": "Last  frame of the output video."}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Smooth camera movement between the two frames.",
                }),
                "model": (SEEDANCE_MODELS, {"default": SEEDANCE_MODELS[0]}),
                "ratio": (RATIO_OPTIONS, {"default": "adaptive"}),
                "duration": (DURATION_OPTIONS, {"default": 5}),
                "generate_audio": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": False}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60, "step": 5}),
                "max_wait":      ("INT",  {"default": 600, "min": 60, "max": 3600, "step": 60}),
            },
            "optional": {
                "first_frame_url": ("STRING", {"default": "", "multiline": False,
                                               "tooltip": "Optional HTTP URL override for the first frame."}),
                "last_frame_url":  ("STRING", {"default": "", "multiline": False,
                                               "tooltip": "Optional HTTP URL override for the last frame."}),
            },
        }

    def generate(
        self,
        api_key,
        first_frame,
        last_frame,
        prompt,
        model,
        ratio,
        duration,
        generate_audio,
        watermark,
        poll_interval,
        max_wait,
        first_frame_url="",
        last_frame_url="",
    ):
        key = resolve_api_key(api_key)

        def _resolve(tensor, url_override):
            if url_override and url_override.strip().startswith("http"):
                return url_override.strip()
            return tensor_to_base64(tensor)

        first_url = _resolve(first_frame, first_frame_url)
        last_url  = _resolve(last_frame,  last_frame_url)

        payload = {
            "model": model,
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": first_url},
                    "role": "first_frame",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": last_url},
                    "role": "last_frame",
                },
            ],
            "ratio": ratio,
            "duration": duration,
            "generate_audio": generate_audio,
            "watermark": watermark,
        }
        video_url, video_path = _run_task(key, payload, poll_interval, max_wait)
        return (video_url, video_path)


# ──────────────────────────────────────────────────────────────────
# Node 4 – Image-to-Video  (reference image)
# ──────────────────────────────────────────────────────────────────

class SeedanceI2VReference:
    """
    Generate a video guided by one to four *reference* images.

    Reference images inform the visual style, characters, or objects that
    should appear in the generated video — not locked to a specific frame.

    NOTE: This mode uses the dedicated model  seedance-1-0-lite-i2v-250428
    (different from the standard Seedance 2 T2V / first-frame models).
    In your prompt you can reference each image by its position tag:
    [Image 1], [Image 2], [Image 3], [Image 4].
    """

    CATEGORY    = "Seedance/Video Generation"
    FUNCTION    = "generate"
    RETURN_TYPES  = ("STRING", "STRING")
    RETURN_NAMES  = ("video_url", "video_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "reference_image_1": ("IMAGE", {"tooltip": "Primary reference image ([Image 1] in prompt)."}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": (
                        "A boy wearing glasses and a blue T-shirt from [Image 1] "
                        "and a corgi dog from [Image 2], sitting on the lawn, in 3D cartoon style"
                    ),
                    "tooltip": "Reference each image with [Image 1], [Image 2], etc.",
                }),
                # Reference I2V uses its own dedicated model
                "model": (SEEDANCE_REFERENCE_MODELS, {"default": SEEDANCE_REFERENCE_MODELS[0]}),
                "ratio": (RATIO_OPTIONS, {"default": "16:9"}),
                "duration": (DURATION_OPTIONS, {"default": 5}),
                "generate_audio": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": False}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60, "step": 5}),
                "max_wait":      ("INT",  {"default": 600, "min": 60, "max": 3600, "step": 60}),
            },
            "optional": {
                # The API supports 1-4 reference images
                "reference_image_2": ("IMAGE", {"tooltip": "Second reference image ([Image 2] in prompt)."}),
                "reference_image_3": ("IMAGE", {"tooltip": "Third  reference image ([Image 3] in prompt)."}),
                "reference_image_4": ("IMAGE", {"tooltip": "Fourth reference image ([Image 4] in prompt)."}),
                "ref_url_1": ("STRING", {"default": "", "multiline": False,
                                         "tooltip": "Optional HTTP URL override for reference image 1."}),
                "ref_url_2": ("STRING", {"default": "", "multiline": False,
                                         "tooltip": "Optional HTTP URL override for reference image 2."}),
                "ref_url_3": ("STRING", {"default": "", "multiline": False,
                                         "tooltip": "Optional HTTP URL override for reference image 3."}),
                "ref_url_4": ("STRING", {"default": "", "multiline": False,
                                         "tooltip": "Optional HTTP URL override for reference image 4."}),
            },
        }

    def generate(
        self,
        api_key,
        reference_image_1,
        prompt,
        model,
        ratio,
        duration,
        generate_audio,
        watermark,
        poll_interval,
        max_wait,
        reference_image_2=None,
        reference_image_3=None,
        reference_image_4=None,
        ref_url_1="",
        ref_url_2="",
        ref_url_3="",
        ref_url_4="",
    ):
        key = resolve_api_key(api_key)

        def _resolve(tensor, url_override):
            if url_override and url_override.strip().startswith("http"):
                return url_override.strip()
            if tensor is None:
                return None
            return tensor_to_base64(tensor)

        content = [{"type": "text", "text": prompt}]

        refs = [
            (reference_image_1, ref_url_1),
            (reference_image_2, ref_url_2),
            (reference_image_3, ref_url_3),
            (reference_image_4, ref_url_4),
        ]
        for tensor, url_override in refs:
            resolved = _resolve(tensor, url_override)
            if resolved:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": resolved},
                    "role": "reference_image",   # ← correct role per API docs
                })

        if len(content) == 1:
            raise ValueError(
                "[Seedance] At least one reference image (or URL) must be provided."
            )

        payload = {
            "model": model,
            "content": content,
            "ratio": ratio,
            "duration": duration,
            "generate_audio": generate_audio,
            "watermark": watermark,
        }
        video_url, video_path = _run_task(key, payload, poll_interval, max_wait)
        return (video_url, video_path)


# ──────────────────────────────────────────────────────────────────
# Registration map  (imported by __init__.py)
# ──────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "SeedanceTextToVideo":         SeedanceTextToVideo,
    "SeedanceI2VFirstFrame":       SeedanceI2VFirstFrame,
    "SeedanceI2VFirstLastFrame":   SeedanceI2VFirstLastFrame,
    "SeedanceI2VReference":        SeedanceI2VReference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedanceTextToVideo":         "Seedance – Text to Video",
    "SeedanceI2VFirstFrame":       "Seedance – Image to Video (First Frame)",
    "SeedanceI2VFirstLastFrame":   "Seedance – Image to Video (First + Last Frame)",
    "SeedanceI2VReference":        "Seedance – Image to Video (Reference)",
}
