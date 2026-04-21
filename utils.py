import base64
import io
import os
import time
import wave
import requests
import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None


# ------------------------------------------------------------------
# ComfyUI IMAGE tensor  →  base64 data-URI
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


def tensor_batch_to_base64_list(tensor) -> list:
    """
    Convert a ComfyUI IMAGE tensor  (B, H, W, C)  to a list of PNG
    base64 data-URI strings, one per image in the batch.
    """
    if len(tensor.shape) == 3:
        return [tensor_to_base64(tensor)]
    return [tensor_to_base64(tensor[i]) for i in range(tensor.shape[0])]


# ------------------------------------------------------------------
# Image download / URL → IMAGE tensor
# ------------------------------------------------------------------

def url_to_image_tensor(url: str):
    """
    Download a single image URL and return a ComfyUI IMAGE tensor
    of shape (1, H, W, C), float32 in [0, 1].
    """
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    if torch is not None:
        return torch.from_numpy(arr).unsqueeze(0)
    return arr[np.newaxis, ...]


def urls_to_image_batch(urls: list):
    """
    Download a list of image URLs and return a ComfyUI IMAGE batch
    tensor (N, H, W, C). If the returned images have different sizes,
    all are resized to match the first one.
    """
    if not urls:
        raise RuntimeError("[Seedream] No image URLs to download.")

    pil_images = []
    for u in urls:
        resp = requests.get(u, timeout=120)
        resp.raise_for_status()
        pil_images.append(Image.open(io.BytesIO(resp.content)).convert("RGB"))

    target_size = pil_images[0].size  # (W, H)
    arrays = []
    for img in pil_images:
        if img.size != target_size:
            img = img.resize(target_size, Image.LANCZOS)
        arrays.append(np.array(img).astype(np.float32) / 255.0)

    batch = np.stack(arrays, axis=0)  # (N, H, W, C)
    if torch is not None:
        return torch.from_numpy(batch)
    return batch


def download_image(url: str, output_dir: str, prefix: str = "seedream",
                   ext: str = "png") -> str:
    """
    Download an image from *url* and save it in *output_dir*.
    Returns the absolute path of the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time() * 1000)
    filename  = f"{prefix}_{timestamp}.{ext}"
    filepath  = os.path.join(output_dir, filename)

    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(filepath, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return filepath


# ------------------------------------------------------------------
# ComfyUI AUDIO dict  →  base64 data-URI
# ------------------------------------------------------------------

def audio_to_base64(audio: dict) -> str:
    """
    Convert a ComfyUI AUDIO dict  {"waveform": Tensor(B,C,S), "sample_rate": int}
    to a WAV base64 data-URI suitable for the API's audio_url field.

    Uses only stdlib (wave module) — no extra dependencies.
    """
    waveform    = audio["waveform"]       # (batch, channels, samples)
    sample_rate = int(audio["sample_rate"])

    if hasattr(waveform, "cpu"):
        waveform = waveform.cpu().numpy()
    else:
        waveform = np.asarray(waveform)

    waveform = waveform[0]                # (channels, samples)
    channels = waveform.shape[0]

    # (channels, samples) → (samples, channels) → int16
    waveform_t    = waveform.T            # (samples, channels)
    waveform_i16  = (waveform_t * 32767).clip(-32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)                # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(waveform_i16.tobytes())

    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:audio/wav;base64,{b64}"


# ------------------------------------------------------------------
# ComfyUI VIDEO object / path  →  base64 data-URI
# ------------------------------------------------------------------

def video_to_base64(video) -> str:
    """
    Convert a ComfyUI VIDEO object (VideoFromFile) or a plain file-path string
    to a base64 data-URI suitable for the API's video_url field.

    Warns if the file exceeds 50 MB, as large payloads may be rejected by the API.
    """
    if isinstance(video, str):
        path = video
    elif hasattr(video, "video_path"):
        path = video.video_path
    elif hasattr(video, "path"):
        path = video.path
    else:
        raise ValueError(f"[Seedance] Cannot extract file path from video object: {type(video)}")

    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > 50:
        print(
            f"[Seedance] Warning: reference video is {size_mb:.1f} MB — "
            "large payloads may exceed API limits. Consider using a URL instead."
        )

    ext  = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"mp4": "video/mp4", "webm": "video/webm", "mov": "video/quicktime"}.get(ext, "video/mp4")

    with open(path, "rb") as f:
        data = f.read()

    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


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
    filename  = f"{prefix}_{timestamp}.mp4"
    filepath  = os.path.join(output_dir, filename)

    print(f"[Seedance] Downloading video → {filepath}")
    resp = requests.get(video_url, stream=True, timeout=120)
    resp.raise_for_status()

    with open(filepath, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"[Seedance] Video saved: {filepath}")
    return filepath


# ------------------------------------------------------------------
# Frame extraction
# ------------------------------------------------------------------

def extract_last_frame(video_path: str):
    """
    Extract the last frame of a video file and return it as a ComfyUI
    IMAGE tensor  (1, H, W, C)  float32 in [0, 1].
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
        raise RuntimeError(f"[Seedance] Could not read last frame from: {video_path}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    arr       = frame_rgb.astype(np.float32) / 255.0  # (H, W, C)

    if torch is not None:
        return torch.from_numpy(arr).unsqueeze(0)   # (1, H, W, C)
    return arr[np.newaxis, ...]


def extract_all_frames(video_path: str):
    """
    Extract every frame of a video and return as a ComfyUI IMAGE batch
    tensor  (N, H, W, C)  float32 in [0, 1].
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[Seedance] Could not open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb.astype(np.float32) / 255.0)
    cap.release()

    if not frames:
        raise RuntimeError(f"[Seedance] No frames extracted from: {video_path}")

    arr = np.stack(frames, axis=0)  # (N, H, W, C)
    if torch is not None:
        return torch.from_numpy(arr)
    return arr


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
