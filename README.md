# comfyui-seedance-nodes

ComfyUI custom nodes for **BytePlus / ByteDance Seedance 2** video-generation API.

---

## Nodes

| Node | Category | Description |
|------|----------|-------------|
| **Seedance – Text to Video** | `Seedance/Video Generation` | Generate a video from a text prompt only. |
| **Seedance – Image to Video (First Frame)** | `Seedance/Video Generation` | Animate an image forward using a text description. The image becomes the first frame. |
| **Seedance – Image to Video (First + Last Frame)** | `Seedance/Video Generation` | Interpolate motion between two images. Supply the start and end frames; the model fills in the middle. |
| **Seedance – Image to Video (Reference)** | `Seedance/Video Generation` | Use one to three reference images to guide the visual style/subject of the video (no frame-position locking). |

All nodes output four values:

| Output | Type | Description |
|--------|------|-------------|
| `video` | VIDEO (STRING fallback) | Native ComfyUI VIDEO object — connect directly to **Save Video** or any video preview node. Falls back to the file path STRING on older ComfyUI builds. |
| `last_frame` | IMAGE | The final frame of the generated video as a ComfyUI IMAGE tensor. Connect to the next node's `first_frame` input to chain clips seamlessly. |
| `video_url` | STRING | CDN URL returned by the API (valid for 48 hours). |
| `video_path` | STRING | Local path where the `.mp4` was saved (inside ComfyUI's output directory). |

---

## Installation

### 1. Clone into custom_nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/comfyui-seedance-nodes.git
```

### 2. Install dependencies

```bash
pip install -r ComfyUI/custom_nodes/comfyui-seedance-nodes/requirements.txt
```

> **Note:** `opencv-python` is required to extract the last frame from generated videos. It is included in `requirements.txt`.

### 3. Configure your API key

Option A – environment variable (recommended):
```bash
export ARK_API_KEY="your-api-key-here"
```

Option B – paste the key directly into the `api_key` field in any node inside ComfyUI.

Get your key at: <https://console.byteplus.com/ark/region:ark+ap-southeast-1/apikey>

### 4. Restart ComfyUI

The nodes will appear under the **Seedance / Video Generation** category in the node browser.

---

## Node reference

### Common inputs (all nodes)

| Input | Type | Description |
|-------|------|-------------|
| `api_key` | STRING | BytePlus ARK API key. If empty, falls back to the `ARK_API_KEY` environment variable. |
| `model` | COMBO | Model ID to use. Currently `dreamina-seedance-2-0-260128`. |
| `ratio` | COMBO | Output aspect ratio: `16:9`, `9:16`, `1:1`, `4:3`, `3:4`, `adaptive`. |
| `duration` | COMBO | Video length in seconds (`5` or `10`). |
| `generate_audio` | BOOLEAN | Whether the model generates ambient audio. |
| `watermark` | BOOLEAN | Whether to add a BytePlus watermark. |
| `poll_interval` | INT | Seconds between status-check calls (default 10). |
| `max_wait` | INT | Maximum total wait time in seconds (default 600). |

### Text to Video – extra inputs

| Input | Description |
|-------|-------------|
| `prompt` | The text description of the video. |

### Image to Video (First Frame) – extra inputs

| Input | Description |
|-------|-------------|
| `first_frame` | ComfyUI IMAGE tensor used as the first frame. |
| `prompt` | Describes how the image should be animated. |
| `image_url_override` | Optional: paste an HTTP URL to use instead of the IMAGE input. |

### Image to Video (First + Last Frame) – extra inputs

| Input | Description |
|-------|-------------|
| `first_frame` | IMAGE tensor – first frame. |
| `last_frame` | IMAGE tensor – last frame. |
| `prompt` | Describes the motion between the two frames. |
| `first_frame_url` / `last_frame_url` | Optional HTTP URL overrides. |

### Image to Video (Reference) – extra inputs

| Input | Description |
|-------|-------------|
| `reference_image_1` | Primary reference IMAGE tensor (required). |
| `reference_image_2` / `reference_image_3` | Additional reference images (optional). |
| `prompt` | Text description of the desired video. |
| `ref_url_1/2/3` | Optional HTTP URL overrides for each reference image. |

---

## Tips

- **Cold-start latency** – the API queues tasks; `poll_interval=10` with `max_wait=600` works for most generations. Increase `max_wait` for 10-second videos at high resolution.
- **Image inputs** – connect any ComfyUI IMAGE node (LoadImage, VAE Decode output, etc.) directly to the image inputs. The node converts the tensor to a PNG base64 data-URI automatically.
- **URL overrides** – if your image is already hosted (e.g., on S3 or a CDN), paste the URL into the `*_url_override` / `*_url_*` fields to skip the base64 encoding step.
- **Output** – videos are saved in `ComfyUI/output/` with a timestamp filename (`seedance_<timestamp>.mp4`). The `video` output connects directly to **Save Video**; the `last_frame` IMAGE output connects to the next node's `first_frame` to chain multiple clips together.

---

## License

MIT
