"""
comfyui-seedance-nodes  –  nodes.py
====================================
ComfyUI custom nodes for BytePlus / ByteDance Seedance 2 video-generation API.

Nodes
-----
SeedanceVideoGenerator       – Unified node (auto-detects mode from inputs)
SeedanceTextToVideo          – Text → Video
SeedanceI2VFirstFrame        – Image-to-Video  (first frame)
SeedanceI2VFirstLastFrame    – Image-to-Video  (first + last frame)
SeedanceI2VReference         – Image-to-Video  (reference image style)
"""

import io
import os

import numpy as np
import requests

try:
    import torch
except ImportError:
    torch = None

try:
    import folder_paths
    OUTPUT_DIR = folder_paths.get_output_directory()
except ImportError:
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

from .api_client import SeedanceAPIClient
from .utils import tensor_to_base64, download_video, extract_last_frame, extract_all_frames, resolve_api_key

# ──────────────────────────────────────────────────────────────────
# VIDEO type detection – requires ComfyUI with comfy_api available
# ──────────────────────────────────────────────────────────────────

try:
    from comfy_api.input_impl.video_types import VideoFromFile as _VideoFromFile
    _VIDEO_TYPE = "VIDEO"

    def _make_video(path: str):
        return _VideoFromFile(path)

except ImportError:
    _VIDEO_TYPE = "STRING"

    def _make_video(path: str):
        return path


# ──────────────────────────────────────────────────────────────────
# Shared constants
# ──────────────────────────────────────────────────────────────────

SEEDANCE_MODELS = [
    "dreamina-seedance-2-0-260128",
]

SEEDANCE_REFERENCE_MODELS = [
    "seedance-1-0-lite-i2v-250428",
    "dreamina-seedance-2-0-260128",
]

ALL_MODELS = [
    "dreamina-seedance-2-0-260128",
    "seedance-1-0-lite-i2v-250428",
]

RATIO_OPTIONS      = ["16:9", "9:16", "1:1", "4:3", "3:4", "adaptive"]
DURATION_OPTIONS   = [5, 10, 15]
RESOLUTION_OPTIONS = ["default", "480p", "720p"]


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _url_to_image_tensor(url: str):
    """Download an image URL and return a ComfyUI IMAGE tensor (1, H, W, C)."""
    from PIL import Image
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0  # (H, W, C)
    if torch is not None:
        return torch.from_numpy(arr).unsqueeze(0)
    return arr[np.newaxis, ...]


def _run_task(api_key: str, payload: dict, poll_interval: int, max_wait: int):
    """
    Create a task, poll until done, download the video.

    Returns: (video, last_frame, all_frames, video_url, video_path)
      - video      : VideoFromFile object (or path STRING as fallback)
      - last_frame : IMAGE tensor (1, H, W, C) – from API if return_last_frame
                     was requested, otherwise extracted with OpenCV
      - all_frames : IMAGE tensor (N, H, W, C) – every frame of the video
      - video_url  : CDN URL string
      - video_path : local .mp4 path string
    """
    client  = SeedanceAPIClient(api_key)
    task_id = client.create_task(payload)
    print(f"[Seedance] Task created: {task_id}")
    result = client.poll_task(task_id, poll_interval=poll_interval, max_wait=max_wait)

    content   = result.get("content", {})
    video_url = content.get("video_url", "")
    if not video_url:
        raise RuntimeError(f"[Seedance] No video_url in response: {result}")

    video_path = download_video(video_url, OUTPUT_DIR, prefix="seedance")
    video      = _make_video(video_path)
    all_frames = extract_all_frames(video_path)

    # Prefer the API-returned last frame (PNG, no watermark) when available
    api_last_frame_url = (
        content.get("last_frame_image_url")
        or content.get("last_frame_url")
        or content.get("last_frame_image")
        or ""
    )
    if api_last_frame_url:
        print(f"[Seedance] Using API-provided last frame: {api_last_frame_url}")
        try:
            last_frame = _url_to_image_tensor(api_last_frame_url)
        except Exception as exc:
            print(f"[Seedance] Could not download API last frame ({exc}), falling back to OpenCV.")
            last_frame = extract_last_frame(video_path)
    else:
        last_frame = extract_last_frame(video_path)

    return video, last_frame, all_frames, video_url, video_path


def _apply_resolution(payload: dict, resolution: str) -> None:
    if resolution and resolution != "default":
        payload["resolution"] = resolution


def _append_reference_media(content: list, video_url: str, audio_url: str) -> None:
    """
    Append reference_video and/or reference_audio entries to the content array.
    Reference them in the prompt as [Video 1] and [Audio 1] respectively.
    """
    if video_url and video_url.strip().startswith("http"):
        content.append({
            "type":      "video_url",
            "role":      "reference_video",
            "video_url": {"url": video_url.strip()},
        })
    if audio_url and audio_url.strip().startswith("http"):
        content.append({
            "type":      "audio_url",
            "role":      "reference_audio",
            "audio_url": {"url": audio_url.strip()},
        })


# Shared optional inputs present on every node
_MEDIA_REF_OPTIONAL = {
    "reference_video_url": ("STRING", {
        "default":   "",
        "multiline": False,
        "tooltip":   "Optional HTTP URL of a reference video. Reference it in your prompt as [Video 1].",
    }),
    "reference_audio_url": ("STRING", {
        "default":   "",
        "multiline": False,
        "tooltip":   "Optional HTTP URL of a reference audio track. Reference it in your prompt as [Audio 1].",
    }),
}


# ──────────────────────────────────────────────────────────────────
# Unified Node – auto-detects mode from connected inputs
# ──────────────────────────────────────────────────────────────────

class SeedanceVideoGenerator:
    """
    Single unified Seedance node. Mode is auto-detected from connected inputs:

      • No images connected           → Text-to-Video
      • first_frame only              → Image-to-Video (first frame)
      • first_frame + last_frame      → Image-to-Video (first + last frame)
      • reference_image_1 (+ 2/3/4)  → Image-to-Video (reference images)

    All modes accept an optional reference video ([Video 1] in prompt),
    reference audio ([Audio 1] in prompt), and return_last_frame.

    For reference-image mode use model seedance-1-0-lite-i2v-250428 and
    reference each image in your prompt with [Image 1], [Image 2], etc.
    """

    CATEGORY     = "Seedance/Video Generation"
    FUNCTION     = "generate"
    RETURN_TYPES = (_VIDEO_TYPE, "IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("video", "last_frame", "frames", "video_url", "video_path")

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
                    "tooltip": (
                        "Text description of the video. "
                        "Use [Image 1-4] for reference images, "
                        "[Video 1] for reference video, [Audio 1] for reference audio."
                    ),
                }),
                "model": (ALL_MODELS, {
                    "default": ALL_MODELS[0],
                    "tooltip": (
                        "dreamina-seedance-2-0-260128 → T2V / first-frame / first+last modes. "
                        "seedance-1-0-lite-i2v-250428 → reference image mode."
                    ),
                }),
                "ratio":           (RATIO_OPTIONS,      {"default": "16:9"}),
                "duration":        (DURATION_OPTIONS,   {"default": 5}),
                "resolution":      (RESOLUTION_OPTIONS, {
                    "default": "default",
                    "tooltip": "Output resolution. 'default' lets the API decide.",
                }),
                "generate_audio":  ("BOOLEAN", {"default": False}),
                "watermark":       ("BOOLEAN", {"default": False}),
                "return_last_frame": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Ask the API to return a PNG of the last frame (no watermark). "
                        "Useful for chaining consecutive videos: connect last_frame → first_frame of the next node."
                    ),
                }),
                "poll_interval": ("INT", {"default": 10, "min": 5,  "max": 60,   "step": 5,
                                          "tooltip": "Seconds between status-check requests."}),
                "max_wait":      ("INT", {"default": 600, "min": 60, "max": 3600, "step": 60,
                                          "tooltip": "Maximum seconds to wait for the task."}),
            },
            "optional": {
                # ── I2V first-frame / first+last frame ───────────────
                "first_frame": ("IMAGE", {
                    "tooltip": (
                        "Connect to enable I2V mode. "
                        "Alone → first-frame mode. With last_frame → interpolation mode."
                    ),
                }),
                "last_frame": ("IMAGE", {
                    "tooltip": "Connect together with first_frame to enable first+last interpolation mode.",
                }),
                "first_frame_url": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Optional HTTP URL override for first_frame.",
                }),
                "last_frame_url": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Optional HTTP URL override for last_frame.",
                }),
                # ── Reference images ──────────────────────────────────
                "reference_image_1": ("IMAGE", {
                    "tooltip": "Connect to enable reference mode ([Image 1] in prompt).",
                }),
                "reference_image_2": ("IMAGE", {
                    "tooltip": "Optional second reference image ([Image 2] in prompt).",
                }),
                "reference_image_3": ("IMAGE", {
                    "tooltip": "Optional third reference image ([Image 3] in prompt).",
                }),
                "reference_image_4": ("IMAGE", {
                    "tooltip": "Optional fourth reference image ([Image 4] in prompt).",
                }),
                "ref_url_1": ("STRING", {"default": "", "multiline": False,
                                         "tooltip": "Optional HTTP URL override for reference image 1."}),
                "ref_url_2": ("STRING", {"default": "", "multiline": False,
                                         "tooltip": "Optional HTTP URL override for reference image 2."}),
                "ref_url_3": ("STRING", {"default": "", "multiline": False,
                                         "tooltip": "Optional HTTP URL override for reference image 3."}),
                "ref_url_4": ("STRING", {"default": "", "multiline": False,
                                         "tooltip": "Optional HTTP URL override for reference image 4."}),
                # ── Reference video / audio ───────────────────────────
                **_MEDIA_REF_OPTIONAL,
            },
        }

    def generate(
        self,
        api_key,
        prompt,
        model,
        ratio,
        duration,
        resolution,
        generate_audio,
        watermark,
        return_last_frame,
        poll_interval,
        max_wait,
        first_frame=None,
        last_frame=None,
        first_frame_url="",
        last_frame_url="",
        reference_image_1=None,
        reference_image_2=None,
        reference_image_3=None,
        reference_image_4=None,
        ref_url_1="",
        ref_url_2="",
        ref_url_3="",
        ref_url_4="",
        reference_video_url="",
        reference_audio_url="",
    ):
        key = resolve_api_key(api_key)

        def _resolve(tensor, url_override):
            if url_override and url_override.strip().startswith("http"):
                return url_override.strip()
            if tensor is None:
                return None
            return tensor_to_base64(tensor)

        has_ref1  = reference_image_1 is not None or (ref_url_1 and ref_url_1.strip().startswith("http"))
        has_first = first_frame is not None        or (first_frame_url and first_frame_url.strip().startswith("http"))
        has_last  = last_frame is not None         or (last_frame_url  and last_frame_url.strip().startswith("http"))

        if has_ref1:
            mode = "reference"
        elif has_first and has_last:
            mode = "first_last"
        elif has_first:
            mode = "first_frame"
        else:
            mode = "t2v"

        print(f"[Seedance] Mode detected: {mode}")

        content = [{"type": "text", "text": prompt}]

        if mode == "first_frame":
            img_url = _resolve(first_frame, first_frame_url)
            content.append({"type": "image_url", "image_url": {"url": img_url}})

        elif mode == "first_last":
            first_url = _resolve(first_frame, first_frame_url)
            last_url  = _resolve(last_frame,  last_frame_url)
            content.append({"type": "image_url", "image_url": {"url": first_url}, "role": "first_frame"})
            content.append({"type": "image_url", "image_url": {"url": last_url},  "role": "last_frame"})

        elif mode == "reference":
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
                        "type":      "image_url",
                        "image_url": {"url": resolved},
                        "role":      "reference_image",
                    })

        _append_reference_media(content, reference_video_url, reference_audio_url)

        payload = {
            "model":             model,
            "content":           content,
            "ratio":             ratio,
            "duration":          duration,
            "generate_audio":    generate_audio,
            "watermark":         watermark,
            "return_last_frame": return_last_frame,
        }
        _apply_resolution(payload, resolution)

        video, last_frame_out, all_frames, video_url, video_path = _run_task(key, payload, poll_interval, max_wait)
        return (video, last_frame_out, all_frames, video_url, video_path)


# ──────────────────────────────────────────────────────────────────
# Node 1 – Text to Video
# ──────────────────────────────────────────────────────────────────

class SeedanceTextToVideo:
    """Generate a video from a text prompt using the Seedance 2 API (T2V mode)."""

    CATEGORY     = "Seedance/Video Generation"
    FUNCTION     = "generate"
    RETURN_TYPES = (_VIDEO_TYPE, "IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("video", "last_frame", "frames", "video_url", "video_path")

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
                    "tooltip": "Use [Video 1] / [Audio 1] to reference media.",
                }),
                "model":           (SEEDANCE_MODELS,    {"default": SEEDANCE_MODELS[0]}),
                "ratio":           (RATIO_OPTIONS,      {"default": "16:9"}),
                "duration":        (DURATION_OPTIONS,   {"default": 5}),
                "resolution":      (RESOLUTION_OPTIONS, {
                    "default": "default",
                    "tooltip": "Output resolution. 'default' lets the API decide.",
                }),
                "generate_audio":  ("BOOLEAN", {"default": False}),
                "watermark":       ("BOOLEAN", {"default": False}),
                "return_last_frame": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Ask the API to return a PNG of the last frame (no watermark). "
                        "Useful for chaining consecutive videos."
                    ),
                }),
                "poll_interval": ("INT", {"default": 10, "min": 5,  "max": 60,   "step": 5,
                                          "tooltip": "Seconds between status-check requests."}),
                "max_wait":      ("INT", {"default": 600, "min": 60, "max": 3600, "step": 60,
                                          "tooltip": "Maximum seconds to wait for the task."}),
            },
            "optional": {
                **_MEDIA_REF_OPTIONAL,
            },
        }

    def generate(
        self,
        api_key,
        prompt,
        model,
        ratio,
        duration,
        resolution,
        generate_audio,
        watermark,
        return_last_frame,
        poll_interval,
        max_wait,
        reference_video_url="",
        reference_audio_url="",
    ):
        key     = resolve_api_key(api_key)
        content = [{"type": "text", "text": prompt}]
        _append_reference_media(content, reference_video_url, reference_audio_url)

        payload = {
            "model":             model,
            "content":           content,
            "ratio":             ratio,
            "duration":          duration,
            "generate_audio":    generate_audio,
            "watermark":         watermark,
            "return_last_frame": return_last_frame,
        }
        _apply_resolution(payload, resolution)
        video, last_frame, all_frames, video_url, video_path = _run_task(key, payload, poll_interval, max_wait)
        return (video, last_frame, all_frames, video_url, video_path)


# ──────────────────────────────────────────────────────────────────
# Node 2 – Image-to-Video  (first frame)
# ──────────────────────────────────────────────────────────────────

class SeedanceI2VFirstFrame:
    """
    Generate a video from a first-frame image + text prompt (I2V mode).
    The supplied image is used as the first frame; the model animates it
    forward according to the text description.
    """

    CATEGORY     = "Seedance/Video Generation"
    FUNCTION     = "generate"
    RETURN_TYPES = (_VIDEO_TYPE, "IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("video", "last_frame", "frames", "video_url", "video_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key":     ("STRING", {"default": "", "multiline": False}),
                "first_frame": ("IMAGE",  {
                    "tooltip": "ComfyUI IMAGE tensor used as the first frame of the video.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "The camera slowly zooms out.",
                    "tooltip": "Describes how the image should be animated. Use [Video 1] / [Audio 1] to reference media.",
                }),
                "model":           (SEEDANCE_MODELS,    {"default": SEEDANCE_MODELS[0]}),
                "ratio":           (RATIO_OPTIONS,      {"default": "adaptive"}),
                "duration":        (DURATION_OPTIONS,   {"default": 5}),
                "resolution":      (RESOLUTION_OPTIONS, {"default": "default"}),
                "generate_audio":  ("BOOLEAN", {"default": False}),
                "watermark":       ("BOOLEAN", {"default": False}),
                "return_last_frame": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Ask the API to return a PNG of the last frame (no watermark). Useful for chaining consecutive videos.",
                }),
                "poll_interval": ("INT", {"default": 10, "min": 5,  "max": 60,   "step": 5}),
                "max_wait":      ("INT", {"default": 600, "min": 60, "max": 3600, "step": 60}),
            },
            "optional": {
                "image_url_override": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Optional: paste a direct HTTP URL to use instead of the IMAGE input.",
                }),
                **_MEDIA_REF_OPTIONAL,
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
        resolution,
        generate_audio,
        watermark,
        return_last_frame,
        poll_interval,
        max_wait,
        image_url_override="",
        reference_video_url="",
        reference_audio_url="",
    ):
        key = resolve_api_key(api_key)

        img_url = (
            image_url_override.strip()
            if image_url_override and image_url_override.strip().startswith("http")
            else tensor_to_base64(first_frame)
        )

        content = [
            {"type": "text",      "text": prompt},
            {"type": "image_url", "image_url": {"url": img_url}},
        ]
        _append_reference_media(content, reference_video_url, reference_audio_url)

        payload = {
            "model":             model,
            "content":           content,
            "ratio":             ratio,
            "duration":          duration,
            "generate_audio":    generate_audio,
            "watermark":         watermark,
            "return_last_frame": return_last_frame,
        }
        _apply_resolution(payload, resolution)
        video, last_frame, all_frames, video_url, video_path = _run_task(key, payload, poll_interval, max_wait)
        return (video, last_frame, all_frames, video_url, video_path)


# ──────────────────────────────────────────────────────────────────
# Node 3 – Image-to-Video  (first + last frame)
# ──────────────────────────────────────────────────────────────────

class SeedanceI2VFirstLastFrame:
    """
    Generate a video that starts at *first_frame* and ends at *last_frame*.
    Both images are sent with their respective roles so the model interpolates
    the motion between them guided by the text prompt.
    """

    CATEGORY     = "Seedance/Video Generation"
    FUNCTION     = "generate"
    RETURN_TYPES = (_VIDEO_TYPE, "IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("video", "last_frame", "frames", "video_url", "video_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key":     ("STRING", {"default": "", "multiline": False}),
                "first_frame": ("IMAGE",  {"tooltip": "First frame of the output video."}),
                "last_frame":  ("IMAGE",  {"tooltip": "Last  frame of the output video."}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Smooth camera movement between the two frames.",
                    "tooltip": "Use [Video 1] / [Audio 1] to reference media.",
                }),
                "model":           (SEEDANCE_MODELS,    {"default": SEEDANCE_MODELS[0]}),
                "ratio":           (RATIO_OPTIONS,      {"default": "adaptive"}),
                "duration":        (DURATION_OPTIONS,   {"default": 5}),
                "resolution":      (RESOLUTION_OPTIONS, {"default": "default"}),
                "generate_audio":  ("BOOLEAN", {"default": False}),
                "watermark":       ("BOOLEAN", {"default": False}),
                "return_last_frame": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Ask the API to return a PNG of the last frame (no watermark). Useful for chaining consecutive videos.",
                }),
                "poll_interval": ("INT", {"default": 10, "min": 5,  "max": 60,   "step": 5}),
                "max_wait":      ("INT", {"default": 600, "min": 60, "max": 3600, "step": 60}),
            },
            "optional": {
                "first_frame_url": ("STRING", {"default": "", "multiline": False,
                                               "tooltip": "Optional HTTP URL override for the first frame."}),
                "last_frame_url":  ("STRING", {"default": "", "multiline": False,
                                               "tooltip": "Optional HTTP URL override for the last frame."}),
                **_MEDIA_REF_OPTIONAL,
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
        resolution,
        generate_audio,
        watermark,
        return_last_frame,
        poll_interval,
        max_wait,
        first_frame_url="",
        last_frame_url="",
        reference_video_url="",
        reference_audio_url="",
    ):
        key = resolve_api_key(api_key)

        def _resolve(tensor, url_override):
            if url_override and url_override.strip().startswith("http"):
                return url_override.strip()
            return tensor_to_base64(tensor)

        first_url = _resolve(first_frame, first_frame_url)
        last_url  = _resolve(last_frame,  last_frame_url)

        content = [
            {"type": "text",      "text": prompt},
            {"type": "image_url", "image_url": {"url": first_url}, "role": "first_frame"},
            {"type": "image_url", "image_url": {"url": last_url},  "role": "last_frame"},
        ]
        _append_reference_media(content, reference_video_url, reference_audio_url)

        payload = {
            "model":             model,
            "content":           content,
            "ratio":             ratio,
            "duration":          duration,
            "generate_audio":    generate_audio,
            "watermark":         watermark,
            "return_last_frame": return_last_frame,
        }
        _apply_resolution(payload, resolution)
        video, last_frame_out, all_frames, video_url, video_path = _run_task(key, payload, poll_interval, max_wait)
        return (video, last_frame_out, all_frames, video_url, video_path)


# ──────────────────────────────────────────────────────────────────
# Node 4 – Image-to-Video  (reference image)
# ──────────────────────────────────────────────────────────────────

class SeedanceI2VReference:
    """
    Generate a video guided by one to four *reference* images.

    Reference images inform the visual style, characters, or objects that
    should appear in the generated video — not locked to a specific frame.

    NOTE: This mode uses the dedicated model  seedance-1-0-lite-i2v-250428.
    In your prompt reference each image with [Image 1], [Image 2], etc.,
    a reference video with [Video 1], and a reference audio with [Audio 1].
    """

    CATEGORY     = "Seedance/Video Generation"
    FUNCTION     = "generate"
    RETURN_TYPES = (_VIDEO_TYPE, "IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("video", "last_frame", "frames", "video_url", "video_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key":           ("STRING", {"default": "", "multiline": False}),
                "reference_image_1": ("IMAGE",  {"tooltip": "Primary reference image ([Image 1] in prompt)."}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": (
                        "A boy wearing glasses and a blue T-shirt from [Image 1] "
                        "and a corgi dog from [Image 2], sitting on the lawn, in 3D cartoon style"
                    ),
                    "tooltip": (
                        "Use [Image 1-4] for reference images, "
                        "[Video 1] for reference video, [Audio 1] for reference audio."
                    ),
                }),
                "model":           (SEEDANCE_REFERENCE_MODELS, {"default": SEEDANCE_REFERENCE_MODELS[0]}),
                "ratio":           (RATIO_OPTIONS,             {"default": "16:9"}),
                "duration":        (DURATION_OPTIONS,          {"default": 5}),
                "resolution":      (RESOLUTION_OPTIONS,        {"default": "default"}),
                "generate_audio":  ("BOOLEAN", {"default": False}),
                "watermark":       ("BOOLEAN", {"default": False}),
                "return_last_frame": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Ask the API to return a PNG of the last frame (no watermark). Useful for chaining consecutive videos.",
                }),
                "poll_interval": ("INT", {"default": 10, "min": 5,  "max": 60,   "step": 5}),
                "max_wait":      ("INT", {"default": 600, "min": 60, "max": 3600, "step": 60}),
            },
            "optional": {
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
                **_MEDIA_REF_OPTIONAL,
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
        resolution,
        generate_audio,
        watermark,
        return_last_frame,
        poll_interval,
        max_wait,
        reference_image_2=None,
        reference_image_3=None,
        reference_image_4=None,
        ref_url_1="",
        ref_url_2="",
        ref_url_3="",
        ref_url_4="",
        reference_video_url="",
        reference_audio_url="",
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
                    "type":      "image_url",
                    "image_url": {"url": resolved},
                    "role":      "reference_image",
                })

        if len(content) == 1:
            raise ValueError("[Seedance] At least one reference image (or URL) must be provided.")

        _append_reference_media(content, reference_video_url, reference_audio_url)

        payload = {
            "model":             model,
            "content":           content,
            "ratio":             ratio,
            "duration":          duration,
            "generate_audio":    generate_audio,
            "watermark":         watermark,
            "return_last_frame": return_last_frame,
        }
        _apply_resolution(payload, resolution)
        video, last_frame, all_frames, video_url, video_path = _run_task(key, payload, poll_interval, max_wait)
        return (video, last_frame, all_frames, video_url, video_path)


# ──────────────────────────────────────────────────────────────────
# Registration map  (imported by __init__.py)
# ──────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "SeedanceVideoGenerator":      SeedanceVideoGenerator,
    "SeedanceTextToVideo":         SeedanceTextToVideo,
    "SeedanceI2VFirstFrame":       SeedanceI2VFirstFrame,
    "SeedanceI2VFirstLastFrame":   SeedanceI2VFirstLastFrame,
    "SeedanceI2VReference":        SeedanceI2VReference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedanceVideoGenerator":      "Seedance – Video Generator",
    "SeedanceTextToVideo":         "Seedance – Text to Video",
    "SeedanceI2VFirstFrame":       "Seedance – Image to Video (First Frame)",
    "SeedanceI2VFirstLastFrame":   "Seedance – Image to Video (First + Last Frame)",
    "SeedanceI2VReference":        "Seedance – Image to Video (Reference)",
}
