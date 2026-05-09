# llama-openai-server

Minimal OpenAI-compatible LLM inference server backed by `llama.cpp`, targeting AMD GPUs via ROCm/HIP.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/health` | Liveness |
| `GET` | `/v1/health/ready` | Readiness (model loaded) |
| `GET` | `/v1/models` | List loaded model |
| `POST` | `/v1/chat/completions` | OpenAI Chat Completions |
| `POST` | `/v1/completions` | OpenAI Text Completions |

All generation endpoints support both streaming (`"stream": true` → SSE) and non-streaming responses. Response shapes match the OpenAI API specification.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Build llama-cpp-python from source for your GPU (adjust gfx arch as needed)
bash scripts/install-rocm-llama.sh
```

## Configuration

```bash
cp .env.example .env
# Edit .env: set MODEL_PATH (or HF_REPO_ID + HF_MODEL_FILENAME), N_GPU_LAYERS, etc.
```

## Run

```bash
bash start.sh
# or directly:
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `local-model` | Model name returned in API responses |
| `MODEL_PATH` | — | Absolute or relative path to a GGUF file |
| `HF_REPO_ID` | — | HuggingFace repo for auto-download |
| `HF_MODEL_FILENAME` | — | Filename within the HF repo |
| `CLIP_MODEL_PATH` | — | Path to the CLIP mmproj GGUF (enables Gemma 4 vision) |
| `HF_CLIP_REPO_ID` | — | HuggingFace repo for mmproj auto-download |
| `HF_CLIP_FILENAME` | — | Filename of the mmproj within the HF repo |
| `N_GPU_LAYERS` | `80` | Layers to offload to GPU |
| `N_CTX` | `0` | Context window (0 = auto from VRAM) |
| `N_THREADS` | `8` | CPU threads |
| `N_BATCH` | `1024` | Batch size |
| `MLOCK` | `true` | Lock model in RAM |
| `OFFLOAD_KQV` | `true` | Offload KV cache to GPU |
| `FLASH_ATTN` | `false` | Flash attention |
| `ROCM_REQUIRED` | `true` | Fail startup if ROCm not detected |
| `ALLOW_MOCK_BACKEND` | `false` | Use mock backend (no GPU needed) |
| `LOG_LEVEL` | `INFO` | Logging level |

## ROCm note

`llama-cpp-python` must be built from source targeting your GPU architecture. Pre-built wheels do not include ROCm/HIP support for newer GPUs (e.g. gfx1201 / RX 9070 XT).

```bash
CMAKE_ARGS="-DGGML_HIPBLAS=on -DCMAKE_HIP_ARCHITECTURES=gfx1201" \
  FORCE_CMAKE=1 pip install llama-cpp-python --no-binary llama-cpp-python
```
