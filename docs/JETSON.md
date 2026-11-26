# Subgen on NVIDIA Jetson (arm64 + CUDA)

`mccloud/subgen:jetson` runs whisper.cpp with CUDA on NVIDIA Jetson, inside Docker.

## TL;DR

```bash
docker compose -f docker-compose.jetson.yml up -d
# then check the log for: whisper.cpp is using GPU device: ...
```

Host prerequisites:

* JetPack 6 or newer (L4T r36+).
* `nvidia-container-toolkit` configured (`runtime: nvidia`, or the docker
  daemon's `default-runtime` set to `nvidia`).

Status: the image is built and tested by CI on GitHub's arm64 runner, which
has no GPU (see [CI](#ci)). Reports from Jetson owners are welcome.

## Why a special build

1. **Jetson's GPU driver belongs to the host.** The kernel driver is fixed by
   the L4T BSP, and the container's `libcuda.so.1` must match it.
   `nvidia/cuda:*` runtime images bake in `cuda-compat` driver libraries from
   their own build era; on Jetson those shadow the host's and `cuInit` fails
   with `CUDA_ERROR_NOT_SUPPORTED (801)`. The correct pattern is a plain OS
   base image and letting `nvidia-container-toolkit` mount the host driver.
2. **The regular `:cuda` image is amd64 only** and targets desktop/server GPUs.

## The recipe

Built from the same `Dockerfile.whispercpp` as every other image, with these
build arguments (`.github/workflows/build_JETSON.yml`):

| Argument | Value | Why |
|---|---|---|
| `GGML_BACKEND` | `cuda` | whisper.cpp's CUDA backend |
| `BUILD_BASE` | `nvidia/cuda:12.6.3-devel-ubuntu24.04` (arm64) | compiler and CUDA libraries for the build stage only |
| `RUNTIME_BASE` | `ubuntu:24.04` | plain OS, no driver libraries baked in |
| `CUDA_ARCHITECTURES` | `87` | Orin native code, plus PTX that newer GPUs (Thor) compile on first use |
| `BUNDLE_CUDA_LIBS` | `1` | copies `libcudart`, `libcublas` and `libcublasLt` from the build stage next to `whisper-cli` |

Why CUDA 12.6: JetPack 6 ships CUDA 12.6 drivers, so the image's libraries are
no newer than the oldest supported host. Newer hosts (JetPack 7, CUDA 13
driver) run CUDA 12.x userspace through CUDA's forward compatibility.

The faster-whisper Jetson image this replaces (CTranslate2 from the Jetson AI
Lab index) is preserved on the
[`legacy-stable-ts`](https://github.com/McCloudS/subgen/tree/legacy-stable-ts)
branch.

## Runtime env

* `PUID=0` (see compose): on headless L4T, `/dev/nvmap` is `cr--r----- root:video`.
  CUDA's memory manager needs a writable nvmap handle, so CUDA only works as
  root inside the container. With the default `PUID=99` CUDA fails to
  initialise and whisper.cpp runs on the CPU.
* `NVIDIA_VISIBLE_DEVICES=all` with `runtime: nvidia`.
* `CONCURRENT_TRANSCRIPTIONS=1`: Jetson memory is shared between CPU and GPU,
  and each concurrent job loads its own copy of the model.
* `WHISPER_MODEL=large-v3-turbo-q5_0` needs about 1 GB.

## CI

`.github/workflows/build_JETSON.yml` builds the image natively on GitHub's
hosted `ubuntu-24.04-arm` runner for pull requests and pushes to `main` and
`dev`, then:

* checks that `whisper-cli` and `libggml-cuda.so` resolve every library except
  `libcuda.so.1` (which comes from the Jetson host), and
* transcribes a sample clip. Without a GPU the CUDA backend can't load, so this
  runs on the CPU and proves the image works, not CUDA.

Pushes to `main` publish `mccloud/subgen:jetson` and `:<version>-jetson`;
pushes to `dev` publish `:dev-jetson`.

## Verifying CUDA on a Jetson

Start subgen and send it any transcription (a Bazarr request or `/asr`). The log
shows `whisper.cpp is using GPU device: ...` with the Orin/Thor GPU. If it says
the CPU instead, check `PUID=0` and `runtime: nvidia`.
