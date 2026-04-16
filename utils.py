import base64
import io
import os
import time
import requests
import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None


# ------------------------------------------------------------------
# ComfyUI tensor  →  base64 data-URI
# ------------------------------------------------------------------

def tensor_to_base64(tensor) -> str:
    """
    Convert a ComfyUI IMAGE tensor  (B, H, W, C)  or  (H, W, C)
    to a PNG base64 data-URI string suitable for the API's image_url field.
    """
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # first image in the batch

    np_img = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(np_img)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ------------------------------------------------------------------
# Video download helpers
# ------------------------------------------------------------------

def download_video(video_url: str, output_dir: str, prefix: str = "seedance") -> str:
    """
    Download a video from *video_url* and save it in *output_dir*.
    Returns the absolute path of the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    filename = f"{prefix}_{timestamp}.mp4"
    filepath = os.path.join(output_dir, filename)

    print(f"[Seedance] Downloading video → {filepath}")
    resp = requests.get(video_url, stream=True, timeout=120)
    resp.raise_for_status()

    with open(filepath, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"[Seedance] Video saved: {filepath}")
    return filepath


# ------------------------------------------------------------------
# Last-frame extraction
# ------------------------------------------------------------------

def extract_last_frame(video_path: str):
    """
    Extract the last frame of a video file and return it as a ComfyUI
    IMAGE tensor  (1, H, W, C)  float32 in [0, 1].

    Requires opencv-python (cv2).
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[Seedance] Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(
            f"[Seedance] Could not read last frame from: {video_path}"
        )

    # cv2 reads as BGR → convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    arr = frame_rgb.astype(np.float32) / 255.0  # (H, W, C)

    if torch is not None:
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W, C)
    else:
        # Fallback: wrap in a list so ComfyUI still receives something iterable
        tensor = arr[np.newaxis, ...]

    return tensor


# ------------------------------------------------------------------
# Resolve API key (node field  >  env variable)
# ------------------------------------------------------------------

def resolve_api_key(node_key: str) -> str:
    key = node_key.strip() if node_key else ""
    if not key:
        key = os.environ.get("ARK_API_KEY", "")
    if not key:
        raise ValueError(
            "No API key found. Either fill in the api_key field in the node "
            "or set the ARK_API_KEY environment variable."
        )
    return key
