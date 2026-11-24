# Subgen on NVIDIA Jetson (arm64 + CUDA)

subgen's transcription (faster-whisper / CTranslate2) can run with real CUDA
acceleration on NVIDIA Jetson Orin and Thor, inside Docker.

## TL;DR

```bash
# on the Jetson, in this repo:
docker build -f Dockerfile.jetson -t subgen:jetson-cuda .
# then use docker-compose.jetson.yml (TRANSCRIBE_DEVICE=gpu, runtime: nvidia)
```

Host prerequisites:

* JetPack 6 or newer (L4T r36+; validated on r39.2.1 / JetPack 7.2, Orin).
* `nvidia-container-toolkit` configured (`docker run --runtime nvidia hello`
  style access). On many Jetson setups the docker daemon's
  `default-runtime` is already `nvidia`.

## Why a special Dockerfile

Three things make Jetson CUDA containers different from x86:

1. **PyPI `ctranslate2` aarch64 wheels are CPU-only builds.** On x86,
   `pip install faster-whisper` gets you CUDA CT2; on aarch64 the same wheel
   has no CUDA kernels, so `ctranslate2.get_cuda_device_count()` returns 0 and
   subgen silently transcribes on CPU.
2. **`nvidia/cuda:*` and JP6-era (`r36.x cu128`) base images break on newer
   hosts.** They bake in `cuda-compat` driver userspace from their own build
   era; on Jetson the kernel driver is fixed by the L4T BSP and the container's
   `libcuda.so.1` must match the HOST driver. A container that shadows it
   (compat libs) fails `cuInit` with `CUDA_ERROR_NOT_SUPPORTED (801)`.
   The correct pattern is: plain OS base image + let
   `nvidia-container-toolkit` mount the host driver libraries — verified
   working (`cuInit rc=0`, device `Orin`) on L4T r39.2.1 with driver 595.78.
3. **No CUDA-CT2 cp312 wheels exist on PyPI for Jetson at all** (upstream CT2
   targets CUDA 12.x; CUDA 13 support is still in review upstream). The
   Jetson AI Lab index (`pypi.jetson-ai-lab.io`, `jp6/cu129`) publishes native
   `ctranslate2-*-cp312-cp312-linux_aarch64.whl` builds, but those wheels
   contain only the Python bindings and expect `libctranslate2.so.4` to be
   installed system-wide (the jetson-ai-lab container convention).

## The recipe (Dockerfile.jetson)

| Piece | Source | Notes |
|---|---|---|
| base image | `ubuntu:24.04`, pinned by digest (`UBUNTU_IMAGE` build arg) | matches L4T r38/r39 host glibc (noble); no CUDA baked in |
| `libcuda.so.1` (driver) | HOST, via nvidia-container-toolkit | never bake cuda-compat libs into the image |
| `ctranslate2` Python bindings | wheel URL from `pypi.jetson-ai-lab.io/jp6/cu129` (`ctranslate2-4.5.0-cp312-cp312-linux_aarch64.whl`), `CT2_WHEEL_URL` build arg | native Jetson build; URL carries a `#sha256=` fragment, so pip refuses a changed file |
| `libctranslate2.so.4.5.0` | one ~38 MB layer of `dustynv/speaches:r36.4.0-cu128-24.04`, downloaded by blob digest in the `ct2lib` stage (`CT2_LIB_LAYER`, `CT2_LIB_SHA256` build args) | CUDA-enabled CT2 library (NEEDED: `libcublas.so.12`, `libcudnn.so.9`); version must match the bindings wheel. Fetching one layer avoids pulling the ~7 GB image; both the layer and the extracted `.so` are sha256-checked |
| cuBLAS / cuDNN / cudart / nvrtc 12.9 | `pip` wheels `nvidia-cublas-cu12==12.9.3.13`, `nvidia-cudnn-cu12==9.28.1.4`, `nvidia-cuda-runtime-cu12==12.9.79` | NVIDIA publishes official manylinux aarch64 wheels; put on `LD_LIBRARY_PATH` (site-packages `nvidia/*/lib`) |
| torch / torchaudio | `torch==2.9.1`, `torchaudio==2.9.1` (PyPI aarch64 = CPU builds) | subgen only uses `torch.cuda.is_available()`/`empty_cache()`; CPU torch is harmless |
| faster-whisper / stable-ts | `faster-whisper==1.2.1`, `stable-ts-whisperless==2.19.1` | same app stack as upstream requirements.txt |
| av (PyAV) | `av<19` | PyAV 19 removed `av.open(metadata_errors=...)` which faster-whisper 1.2.1 still passes (SYSTRAN/faster-whisper#1589) |
| fastapi | `fastapi==0.115.12` | subgen.py uses `Query(..., enum=[...])`; the `enum` kwarg was removed in newer FastAPI (breaks `subgen.py` at startup) |

Why cu12 userspace on a cu13 host (JetPack 7): the CT2 native build (and CT2
upstream) targets CUDA 12.x. CUDA minor-version forward compatibility lets a
12.9 userspace run on a 595.78 (CUDA 13-capable) kernel driver — the driver
only needs to be >= the userspace. Verified end to end on this device.

Do **not** install `torch` from `download.pytorch.org/whl/cuXXX` on aarch64 —
those indexes don't match this stack; plain PyPI aarch64 wheels are all
subgen needs.

## Version-matching rules

* bindings wheel version == libctranslate2.so version (both 4.5.0 here).
* Every external input is pinned by hash. When bumping one, update the
  matching build arg and its hash together: `UBUNTU_IMAGE` (index digest),
  `CT2_WHEEL_URL` (`#sha256=` fragment; the jetson-ai-lab simple index lists
  it next to each file), and `CT2_LIB_LAYER` / `CT2_LIB_FILE` /
  `CT2_LIB_SHA256` (layer blob digest from the source image manifest, plus
  the hash of the extracted `.so`).
* If you change CT2 to 4.6.0 (bindings exist at jp6/cu129), you also need a
  4.6.0 `libctranslate2.so.4`; either from a matching jetson-ai-lab image or
  by building CT2 from source (CUDA 12 nvcc required — CT2 does not build with
  CUDA 13 as of writing).
* `faster-whisper` requires `ctranslate2<5,>=4.0`; keep that in mind before
  bumping CT2 to 5.x someday.

## Runtime env

* `PUID=0` (see compose): on headless L4T, `/dev/nvmap` is `cr--r----- root:video`
  — CUDA's memory manager needs a writable nvmap handle, so the CUDA path only
  works as root inside the container. With the default `PUID=99` you get
  `NvRmMemInitNvmap failed: error Permission denied` and CT2 silently falls
  back to CPU (`get_cuda_device_count() == 0`).
* `TRANSCRIBE_DEVICE=gpu` (subgen maps `gpu` → `cuda`) — or `cuda` directly.
* `NVIDIA_VISIBLE_DEVICES=all` (or use `--gpus all` / `runtime: nvidia`).
* `CONCURRENT_TRANSCRIPTIONS`: keep at 1 for large-v3 on 16GB Orin (each CUDA
  worker ≈ 3GB unified memory).
* `CLEAR_VRAM_ON_COMPLETE` is a no-op in this image (torch here has no CUDA);
  CT2's own allocator cache is what matters.
* Optional: `CT2_CUDA_ALLOCATOR=cub` variants are generally not needed on
  Jetson.

## CI

`.github/workflows/build_JETSON.yml` builds this image on GitHub's hosted
`ubuntu-24.04-arm` runner for PRs and pushes to `main` that touch it, then runs
an import smoke test (CT2 bindings + `libctranslate2` + pip CUDA libs all
resolve). No GPU is needed to build. CI cannot exercise CUDA itself, so
changes to the CUDA stack still need a check on a real Jetson (below).
A manual run of the workflow with "push" ticked publishes
`mccloud/subgen:jetson`.

## Verifying CUDA inside the container

```bash
docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all subgen:jetson-cuda \
  python3 -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"
# expect: 1
```

## Verified on (this device)

Jetson Orin (nvgpu), L4T r39.2.1 (JetPack 7.2), driver 595.78, docker + nvidia-container-toolkit 1.19.1:

- `ctranslate2.get_cuda_device_count()` → 1 (as root in-container; see PUID=0 above)
- subgen server starts: `Transcribe device: cuda`; `/v1/audio/transcriptions` returns correct SRT/JSON
- Benchmark (jfk.flac, 11s clip, large-v3): GPU fp16 transcribes in **6.9 s** vs **41.3 s** through the previous `mccloud/subgen:cpu` container (~6×)

## Troubleshooting

* `cuInit` / `cudaGetDeviceCount` → `CUDA_ERROR_NOT_SUPPORTED (801)` or
  `Error 803`: container is not getting the HOST driver. Check
  `nvidia-smi` inside the container and `ldconfig -p | grep libcuda` — the
  libcuda version inside must equal the host L4T driver version. Rebuild or
  remove any image that ships cuda-compat.
* `ImportError: libctranslate2.so.4`: bindings installed without the system
  library. Use Dockerfile.jetson (the `ct2lib` stage provides it).
* `libcublas.so.12: cannot open shared object file`: LD_LIBRARY_PATH must
  include the pip `nvidia/*/lib` dirs (already set in the image).
