# Subgen

[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate/?hosted_button_id=SU4QQP6LH5PF6)
<img src="https://raw.githubusercontent.com/McCloudS/subgen/main/icon.png" width="200">

<details>
<summary><strong>Updates:</strong></summary>

24 Nov 2026 (dev): The `dev` images now use **whisper.cpp**, which runs on NVIDIA (CUDA), Intel and AMD GPUs including integrated graphics (Vulkan), and CPU, on amd64 and arm64. The whisper.cpp images don't include faster-whisper; it is still supported for standalone installs (`TRANSCRIBE_BACKEND=faster-whisper`). Fixes found by end-to-end testing: whisper.cpp couldn't read video files from webhooks, `/batch` or folder scans; `/asr` ignored its `output` parameter and always returned SRT (now `txt`, `vtt`, `tsv` and `json` work); Bazarr's raw audio (`encode=false`) failed on whisper.cpp. Subtitles keep hyphens, wrap every line at 42 characters and pace cues like professional subtitles. `WHISPER_THREADS` now sizes itself for whisper.cpp. See [Speed, memory and concurrent transcriptions](#-speed-memory-and-concurrent-transcriptions) and [Upgrading from stable-ts](#-upgrading-from-stable-ts-main).

8 Jun 2026: Dropped `stable-ts-whisperless` (archived upstream). Subgen now drives `faster-whisper` directly with a built-in Netflix-style subtitle segmenter. New tuning variables: `MAX_LINE_LENGTH`, `GAP_SPLIT_SECS`, `VAD_FILTER`. Removed `CUSTOM_REGROUP` and `WORD_LEVEL_HIGHLIGHT` — these were stable-ts-specific and no longer have any effect.

7 Jun 2026: Fixed a bug where files containing only a **forced** embedded subtitle track were incorrectly treated as having full subtitle coverage and skipped. Forced tracks cover only a small fraction of dialogue (typically foreign-language inserts) and should not count as full coverage. Added `IGNORE_FORCED_SUBTITLES` (default `True`) to control this behaviour.

11 Apr 2026: Fixed subtitle timing on files with audio stream offsets (common in Amazon WEB-DL). Whisper ignores silence padding, causing subtitles to be early by the offset amount. Subgen now detects this via ffprobe and compensates automatically when the source video file is accessible. See [Audio Start-Time Offset Fix](#-audio-start-time-offset-fix) for details.

27 Mar 2026: Potentially added ROCm support for AMD GPU/APUs. I don't have anything to test it, so a fair chance it doesn't work at all.  I'm unsure if it will work with AMD APUs.  Image is: `mccloud/subgen:amd`.  It's pretty large right now at ~10gb. In theory, it should see your AMD card the same way it sees any other cuda device. Some light research shows ROCm only 'officially' supports higher end consumer cards and datacenter cards. `HSA_OVERRIDE_GFX_VERSION` can be set to 'trick' your old cards (and maybe APUs) to work, but you'll have to do your own research/googling. 

17 Mar 2026: Added `WEBHOOK_URL_COMPLETED`. When a task finishes, Subgen will send a POST request with a JSON structure.

4 Mar 2026: Reformmated the readme and spend an hour with Gemini trying to format it and clean it up and it essentially only gave me icons for headers.  

Feb 2026: Contributor helped cut the GPU container size in half. Added `ASR_TIMEOUT` as environment variable to timeout ASR endpoint transcriptions after X seconds.

31 Jan 2026: Added the ability to run the container 'rootless', accepts `PUID` and `PGID` as environment variables and _should_ take `user` at a container level (Podman), let me know!.

13 Jan 2026: Probably fixed the runaway memory problems for CPU only. Added `MODEL_CLEANUP_DELAY` which will wait X seconds before purging the model to clear up (V)RAM. This mostly helps with Bazarr or when concurrent transcriptions is 1. Rewrote ASR (Bazarr) queuing so it should respect queing and follow concurrent transcriptions. Also fixed the error when too many Bazarr or ASR requests would start to fail.  

26 Aug 2025: Renamed environment variables to make them slightly easier to understand. Currently maintains backwards compatibility. See https://github.com/McCloudS/subgen/pull/229

12 Aug 2025: Added distil-large-v3.5

7 Feb 2025: Fixed (V)RAM clearing, added PLEX_QUEUE_SEASON, other extraneous fixes or refactorting.  

23 Dec 2025: Added PLEX_QUEUE_NEXT_EPISODE and PLEX_QUEUE_SERIES. Will automatically start generating subtitles for the next episode in your series, or queue the whole series.  

4 Dec 2025: Added more ENV settings: DETECT_LANGUAGE_OFFSET, PREFERRED_AUDIO_LANGUAGES, SKIP_IF_AUDIO_TRACK_IS, ONLY_SKIP_IF_SUBGEN_SUBTITLE, SKIP_UNKNOWN_LANGUAGE, SKIP_IF_LANGUAGE_IS_NOT_SET_BUT_SUBTITLES_EXIST, SHOULD_WHISPER_DETECT_AUDIO_LANGUAGE

30 Nov 2024: Signifcant refactoring and handling by Muisje. Added language code class for more robustness and flexibility and ability to separate audio tracks to make sure you get the one you want. New ENV Variables: SUBTITLE_LANGUAGE_NAMING_TYPE, SKIP_IF_AUDIO_TRACK_IS, PREFERRED_AUDIO_LANGUAGE, SKIP_IF_TO_TRANSCRIBE_SUB_ALREADY_EXIST
There will be some minor hiccups, so please identify them as we work through this major overhaul.

22 Nov 2024: Updated to support large-v3-turbo

30 Sept 2024: Removed webui

5 Sept 2024: Fixed Emby response to a test message/notification. Clarified Emby/Plex/Jellyfin instructions for paths.

14 Aug 2024: Cleaned up usage of kwargs across the board a bit. Added ability for /asr to encode or not, so you don't need to worry about what files/formats you upload.

3 Aug 2024: Added SUBGEN_KWARGS environment variable which allows you to override the model.transcribe with most options you'd like from faster-whisper. This won't be exposed via the webui, it's best to set directly.

21 Apr 2024: Fixed queuing with thanks to https://github.com/xhzhu0628 @ https://github.com/McCloudS/subgen/pull/85. Bazarr intentionally doesn't follow `CONCURRENT_TRANSCRIPTIONS` because it needs a time sensitive response.

31 Mar 2024: Removed `/subsync` endpoint and general refactoring. Open an issue if you were using it!

22 Mar 2024: Added LRC capability via see: `LRC_FOR_AUDIO_FILES | True | Will generate LRC (instead of SRT) files for filetypes: '.mp3', '.flac', '.wav', '.alac', '.ape', '.ogg', '.wma', '.m4a', '.m4b', '.aac', '.aiff'`

21 Mar 2024: Added a 'wizard' into the launcher that will help standalone users get common Bazarr variables configured. See below in Launcher section. Removed 'Transformers' as an option. While I usually don't like to remove features, I don't think anyone is using this and the results are wildly unpredictable and often cause out of memory errors. Added two new environment variables called `USE_MODEL_PROMPT` and `CUSTOM_MODEL_PROMPT`. If `USE_MODEL_PROMPT` is `True` it will use `CUSTOM_MODEL_PROMPT` if set, otherwise will default to using the pre-configured language pairings. These pre-configurated translations are geared towards fixing some audio that may not have punctionation. We can prompt it to try to force the use of punctuation during transcription.

19 Mar 2024: Added a `MONITOR` environment variable. Will 'watch' or 'monitor' your `TRANSCRIBE_FOLDERS` for changes and run on them. Useful if you just want to paste files into a folder and get subtitles.   

5 Mar 2024: Cleaned up logging. Added timestamps option (if Debug = True, timestamps will print in logs).

4 Mar 2024: Updated Dockerfile CUDA to 12.2.2 (From CTranslate2). Added endpoint `/status` to return Subgen version. Can also use distil models now! See variables below!

29 Feb 2024: Changed default port to align with whisper-asr and deconflict other consumers of the previous port.

11 Feb 2024: Added a 'launcher.py' file for Docker to prevent huge image downloads. Now set UPDATE to True if you want pull the latest version, otherwise it will default to what was in the image on build. Docker builds will still be auto-built on any commit. If you don't want to use the auto-update function, no action is needed on your part and continue to update docker images as before. Fixed bug where detect-langauge could return an empty result. Reduced useless debug output that was spamming logs and defaulted DEBUG to True. Added APPEND, which will add a transcribed watermark at the end of a subtitle.

10 Feb 2024: Added some features from JaiZed's branch such as skipping if SDH subtitles are detected, functions updated to also be able to transcribe audio files, allow individual files to be manually transcribed, and a better implementation of forceLanguage. Added `/batch` endpoint (Thanks JaiZed). Allows you to navigate in a browser to http://subgen_ip:9000/docs and call the batch endpoint which can take a file or a folder to manually transcribe files. Added CLEAR_VRAM_ON_COMPLETE, HF_TRANSFORMERS, HF_BATCH_SIZE. Hugging Face Transformers boast '9x increase', but my limited testing shows it's comparable to faster-whisper or slightly slower. I also have an older 8gb GPU. Simplest way to persist HF Transformer models is to set "HF_HUB_CACHE" and set it to "/subgen/models" for Docker (assuming you have the matching volume).

8 Feb 2024: Added FORCE_DETECTED_LANGUAGE_TO to force a wrongly detected language. Fixed asr to actually use the language passed to it.  

5 Feb 2024: General housekeeping, minor tweaks on the TRANSCRIBE_FOLDERS function.

28 Jan 2024: Fixed issue with ffmpeg python module not importing correctly. Removed separate GPU/CPU containers. Also removed the script from installing packages, which should help with odd updates I can't control (from other packages/modules). The image is a couple gigabytes larger, but allows easier maintenance.  

19 Dec 2023: Added the ability for Plex and Jellyfin to automatically update metadata so the subtitles shows up properly on playback. (See https://github.com/McCloudS/subgen/pull/33 from Rikiar73574)  

31 Oct 2023: Added Bazarr support via Whipser provider.

25 Oct 2023: Added Emby (IE http://192.168.1.111:9000/emby) support and TRANSCRIBE_FOLDERS, which will recurse through the provided folders and generate subtitles. It's geared towards attempting to transcribe existing media without using a webhook.

23 Oct 2023: There are now two docker images, ones for CPU (it's smaller): mccloud/subgen:latest, mccloud/subgen:cpu, the other is for cuda/GPU: mccloud/subgen:cuda. I also added Jellyfin support and considerable cleanup in the script. I also renamed the webhooks, so they will require new configuration/updates on your end. Instead of /webhook they are now /plex, /tautulli, and /jellyfin.

22 Oct 2023: The script should have backwards compability with previous envirionment settings, but just to be sure, look at the new options below. If you don't want to manually edit your environment variables, just edit the script manually. While I have added GPU support, I haven't tested it yet.

19 Oct 2023: And we're back! Uses faster-whisper. Shouldn't break anything from previous settings, but adds a couple new options that aren't documented at this point in time. As of now, this is not a docker image on dockerhub. The potential intent is to move this eventually to a pure python script, primarily to simplify my efforts. Quick and dirty to meet dependencies: pip or `pip3 install flask requests faster-whisper`

2 Feb 2023: Added Tautulli webhooks back in. Didn't realize Plex webhooks was PlexPass only. See below for instructions to add it back in.

31 Jan 2023 : Rewrote the script substantially to remove Tautulli and fix some variable handling. For some reason my implementation requires the container to be in host mode. My Plex was giving "401 Unauthorized" when attempt to query from docker subnets during API calls. (**Fixed now, it can be in bridge**)
</details>

---

## 🎬 What is this?
Subgen transcribes your personal media to create subtitles (`.srt` or `.lrc`) from audio/video files. It can transcribe non-English languages to themselves, or translate foreign languages into English. 

It is designed to integrate perfectly with **Bazarr** (as a Whisper Provider), or run via webhooks triggered directly by your **Plex, Emby, Jellyfin, or Tautulli** servers whenever media is added or played. Under the hood, it uses [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (or `faster-whisper`) for AI transcription and a built-in Netflix-style subtitle segmenter for clean, readable output. CPU, NVIDIA (CUDA), and Intel or AMD GPUs (Vulkan) are supported.

## 🤔 Why?
Some shows just won't have subtitles available, or embedded H265 subtitles might be wildly out of sync. This gap-fills everything else by generating highly accurate subtitles locally on your own hardware. 

---

## ⚡ Quick Start: Bazarr (The Bare Minimum)
If you just want to plug Subgen into Bazarr and get going, here is the absolute minimum you need to configure in your Subgen Docker container. **No path mapping or media mounts are needed!**

**1. Set your Environment Variables in Subgen:**
* Pick the image for your hardware (see [Docker](#1-docker-recommended)) and give the container the GPU: `--gpus all` for NVIDIA, `--device /dev/dri` for Intel or AMD. whisper.cpp uses the GPU automatically. (`TRANSCRIBE_DEVICE=cuda` is only needed for faster-whisper.)
* `WHISPER_MODEL`: Default is `medium`. `large-v3-turbo` is faster and more accurate on any GPU; its quantized version `large-v3-turbo-q5_0` needs about 1 GB of video memory. **Note:** `large-v3-turbo` only supports transcription. If you need translation (e.g. `TRANSCRIBE_OR_TRANSLATE=translate`), use `large-v3` instead.
* `CONCURRENT_TRANSCRIPTIONS`: Default is `2`. Each concurrent job holds its own copy of the model with whisper.cpp; see [Speed, memory and concurrent transcriptions](#-speed-memory-and-concurrent-transcriptions).

**2. Configure Bazarr:**
* In Bazarr, go to **Settings > Whisper Provider**.
* Select **Whisper** as the provider.
* Set the **Docker Endpoint** to your Subgen IP and port: `http://<your-ip>:9000` *(Note: Do not use `127.0.0.1` if Bazarr is also in a Docker container).*
* Save! Subgen will now act as an invisible, self-hosted API for Bazarr's transcription requests.

**3. Disable Auto-Sync for Subgen subtitles (important):**
Subgen already produces accurately timed subtitles. If you have Bazarr's **Automatic Subtitles Audio Synchronization** enabled, you must exclude `whisperai` from it — otherwise Bazarr will run ffsubsync on top of already-synced subtitles and degrade their quality.
* In Bazarr, go to **Settings > Subtitles > Audio Synchronization**.
* Under **"Do not sync subtitles downloaded from those providers"**, add **`whisperai`**.

---

## 🔧 Audio Start-Time Offset Fix

Some media containers — particularly Amazon WEB-DL files — have an audio stream that starts later than the video stream (e.g., audio `start_time` of ~4 seconds). When Bazarr extracts audio from these files, it compensates by prepending silence via ffmpeg's `adelay` filter. However, Whisper's speech recognition completely ignores this digital silence, producing timestamps that are early by the offset amount (e.g., every subtitle appears ~4 seconds too early).

Subgen now automatically detects and compensates for this. When the source video file is accessible, it uses `ffprobe` to read the audio stream's `start_time` metadata, then shifts all Whisper timestamps forward by that amount after transcription.

**This fix is fully backwards compatible.** If the video file is not accessible, or has no audio offset (i.e. `start_time` is 0), behaviour is completely unchanged.

### How to enable it (Bazarr)

1. **Mount your media into the Subgen container** with the same paths that Bazarr sees. For example, if Bazarr sees TV shows at `/tv`, add a volume mount so Subgen also sees `/tv`:
   ```yaml
   volumes:
     - /path/to/your/tv:/tv
     - /path/to/your/movies:/movies
   ```

2. **Enable "Pass Video Name" in Bazarr.** Go to **Settings > Whisper Provider** and check the **Pass Video Name** option. This tells Bazarr to send the video file path alongside the audio, allowing Subgen to look up the source file and detect any audio offset.

That's it. No new environment variables are required. Files without an audio offset are unaffected.

---

## 🛠 Installation & Setup

### 1. Docker (Recommended)
The easiest way to run Subgen is via Docker. We maintain an image on Docker Hub (`mccloud/subgen`).
* `mccloud/subgen:latest` (Supports both CPU and GPU/CUDA)
* `mccloud/subgen:cpu` (Smaller image, CPU only)
* `mccloud/subgen:amd` (ROCm/AMD GPU)
* Development builds (whisper.cpp engine, may be unstable):
  * `mccloud/subgen:dev-cuda` (also `:dev`): NVIDIA (CUDA 12.8, GTX 10xx/Tesla P4 and newer). Run with `--gpus all`.
  * `mccloud/subgen:dev-vulkan`: Intel iGPU/Arc, AMD GPUs/APUs, and any other Vulkan GPU. Pass `/dev/dri` into the container. `:dev-whispercpp` and `:dev-amd` are deprecated aliases of this tag; switch to `:dev-vulkan`.
  * `mccloud/subgen:dev-cpu`: CPU only (amd64 and arm64).
  * Tags are named by GPU backend, so pick the one your hardware needs: `cuda` for NVIDIA, `vulkan` for everything else with a GPU, `cpu` for no GPU.

**Crucial Note on Volume Mapping:** If you are using Plex/Emby/Jellyfin/Tautulli webhooks, **Subgen must see your media paths exactly identically to how your media server sees them.** For example, if Plex uses `/Share/media/TV:/tv`, Subgen needs that exact same volume mount. *(Note: This does not apply to Bazarr, which sends audio over HTTP).*

### 1b. NVIDIA Jetson (Orin / Thor, arm64)

The images above are amd64, and pip's `ctranslate2` aarch64 wheels are CPU-only, so Jetson containers silently fall back to CPU. `Dockerfile.jetson` provides a Jetson-specific build with real CUDA acceleration:

> `docker build -f Dockerfile.jetson -t subgen:jetson-cuda .`
> `docker compose -f docker-compose.jetson.yml up -d`

Requirements: JetPack 6+ (L4T r36+) with `nvidia-container-toolkit`, and build on the Jetson itself or any other arm64 machine (no GPU needed to build). Set `TRANSCRIBE_DEVICE=gpu`. See [docs/JETSON.md](docs/JETSON.md) for the full explanation of how CUDA is wired up (and why the usual `nvidia/cuda` base images break on Jetson).

### 2. Standalone (Without Docker)
1. Install Python 3.9–3.11 and `ffmpeg`.
2. Ensure you have the proper NVIDIA drivers/CUDA toolkit installed (if using GPU).
3. Download `launcher.py` from this repository and run:
   > `python3 launcher.py -u -i -s`

*(Launcher includes a wizard to help standalone users easily configure common variables).*

### 3. Unraid
While Unraid doesn't have an app or template for quick install, with minor manual work, you can easily install it. See [this discussion thread](https://github.com/McCloudS/subgen/discussions/137) for pictures and steps.

---

## 🔌 Integrations & Webhooks Setup

Choose your preferred integration below. **Do not enable multiple webhooks for the same media events** (e.g., don't use both Tautulli and Plex webhooks for "playback start"), or you will generate duplicate subtitles!

### 🟠 Plex
Requires Plex Pass. Plex and Subgen must have identical path configurations (or use Path Mapping).
1. In Plex, go to **Settings > Webhooks**.
2. Add a new webhook pointing to your Subgen instance: `http://<your-ip>:9000/plex`
3. You will also need to generate a [Plex Token](https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/).
4. **Relevant Variables:** `PLEX_SERVER`, `PLEX_TOKEN`.

### 🔵 Jellyfin
Jellyfin and Subgen must have identical path configurations (or use Path Mapping).
1. Install the **Webhooks** plugin in Jellyfin.
2. Click **Add Generic Destination**. 
3. Name it whatever you like, and set the Webhook URL to: `http://<your-ip>:9000/jellyfin`
4. Check **Item Added**, **Playback Start**, and **Send All Properties**.
5. Click **Add Request Header**. Set Key: `Content-Type` and Value: `application/json`.
6. **Relevant Variables:** `JELLYFIN_SERVER`, `JELLYFIN_TOKEN`.

### 🟢 Emby
Emby and Subgen must have identical path configurations (or use Path Mapping). Emby responses contain full info, so no API tokens are required!
1. In Emby, create a webhook pointing to: `http://<your-ip>:9000/emby`
2. Set **Request content type** to `multipart/form-data`.
3. Configure your desired events (Usually `New Media Added`, `Start`, and `Unpause`).

### 🟣 Tautulli
Tautulli and Subgen must have identical path configurations (or use Path Mapping).
Create two separate Webhooks in Tautulli pointing to `http://<your-ip>:9000/tautulli` using the **POST** method.

**Webhook 1: Playback Start**
*   **Trigger:** Playback Start
*   **JSON Header:** `{"source": "Tautulli"}`
*   **Data (JSON):** 
    > `{"event": "played", "file": "{file}", "filename": "{filename}", "mediatype": "{media_type}"}`

**Webhook 2: Recently Added**
*   **Trigger:** Recently Added
*   **JSON Header:** `{"source": "Tautulli"}`
*   **Data (JSON):** 
    > `{"event": "added", "file": "{file}", "filename": "{filename}", "mediatype": "{media_type}"}`

---

## ⚙️ Configuration (Environment Variables)

*Note: Subgen recently standardized environment variable names (e.g., `PLEX_TOKEN`). Legacy names (e.g., `PLEXTOKEN`) are still fully supported for backwards compatibility!*

### 🧠 Core Whisper & AI Settings
| Variable | Default | Description |
|---|---|---|
| `TRANSCRIBE_DEVICE` | `cpu` | Device to transcribe on: `cpu`, `gpu`, or `cuda`. |
| `WHISPER_MODEL` | `medium` | Model to use: `tiny`, `base`, `small`, `medium`, `large-v3`, `distil-large-v3`, `large-v3-turbo`, quantized versions like `large-v3-turbo-q5_0`, etc. See [Choosing a model](#-choosing-a-model). **`large-v3-turbo` does not support translation** (fine-tuned on transcription only) — use `large-v3` if `TRANSCRIBE_OR_TRANSLATE=translate`. |
| `CONCURRENT_TRANSCRIPTIONS` | `2` | Number of files to process in parallel. With whisper.cpp each one loads its own copy of the model, so memory grows with this number. See [Speed, memory and concurrent transcriptions](#-speed-memory-and-concurrent-transcriptions). |
| `WHISPER_THREADS` | *auto* / `4` | CPU threads per transcription. whisper.cpp: unset means the CPU cores are shared between `CONCURRENT_TRANSCRIPTIONS` (between 4 and 16 each). faster-whisper: `4`. |
| `COMPUTE_TYPE` | `auto` | Precision quantization mapping (e.g., `float16`, `int8`). See [CTranslate2 docs](https://github.com/OpenNMT/CTranslate2/blob/master/docs/quantization.md). |
| `CLEAR_VRAM_ON_COMPLETE` | `True` | Do garbage collection and clear the model from VRAM when the queue is empty. |
| `MODEL_CLEANUP_DELAY` | `30` | Seconds to wait before clearing the Whisper model from memory. |
| `ASR_TIMEOUT` | `18000` | Seconds to wait before timing out a transcription request (default 5 hours). |
| `VAD_FILTER` | `False` | Run Silero VAD before transcription to strip silence. Can reduce hallucinations on quiet segments, but may also drop dialogue. Set to `True` to enable. **faster-whisper only for now:** whisper.cpp's VAD breaks subtitle timing, so subgen ignores this setting with whisper.cpp and logs a warning. |
| `SUBGEN_KWARGS` | `{}` | JSON dict of extra kwargs passed directly to `faster-whisper`'s `model.transcribe()` (e.g. `{"beam_size": 10}`). For advanced users. |

### 🖥️ Backend & GPU (whisper.cpp)
whisper.cpp runs on NVIDIA (CUDA), AMD and Intel GPUs including integrated graphics (Vulkan), and CPU. It is the default whenever the `whisper-cli` binary is present (the `whispercpp` images).
| Variable | Default | Description |
|---|---|---|
| `TRANSCRIBE_BACKEND` | `auto` | `auto` uses whisper.cpp when `whisper-cli` is installed, otherwise faster-whisper. Force one with `whispercpp` or `faster-whisper`. The whisper.cpp images don't include faster-whisper, so asking for it there logs a warning and uses whisper.cpp. |
| `WHISPER_CPP_MODEL` | *(derived)* | Path to a GGML model file. If unset, it is derived from `WHISPER_MODEL` (e.g. `large-v3-turbo` → `ggml-large-v3-turbo.bin`) and downloaded to `MODEL_PATH`. |
| `WHISPER_CPP_REPO` | *(derived)* | Hugging Face repo to download `WHISPER_CPP_MODEL` from. |
| `WHISPER_CLI_PATH` | `whisper-cli` | Path to the `whisper-cli` binary. |
| `WHISPER_CPP_DEVICE` | *(auto)* | whisper.cpp uses a GPU automatically when it can see one. Set a GPU index (`0`, `1`, ...) to pick one, or `cpu` to force CPU. `TRANSCRIBE_DEVICE` only applies to faster-whisper. Check the log for `whisper.cpp is using GPU device: ...`. |
| `WHISPER_CPP_VAD_MODEL` | *(auto)* | Silero VAD model for whisper.cpp. Currently unused: whisper.cpp VAD is disabled (see `VAD_FILTER`). |
| `VAD_THRESHOLD` | `0.5` | Speech probability threshold for whisper.cpp VAD (currently disabled). |

### ⚡ Processing Triggers & Queuing
*(Not relevant for Bazarr users)*
| Variable | Default | Description |
|---|---|---|
| `PROCESS_ADDED_MEDIA` | `True` | Generate subs for newly added media (when triggered by webhook). |
| `PROCESS_MEDIA_ON_PLAY` | `True` | Generate subs for media when it is played (when triggered by webhook). |
| `TRANSCRIBE_FOLDERS` | `''` | Pipe-separated list (e.g., `/tv&#124;/movies`) to recurse through and queue existing media. |
| `MONITOR` | `False` | Actively watches `TRANSCRIBE_FOLDERS` in real-time for newly pasted files. |
| `SKIP_STARTUP_SCAN` | `False` | Skips the startup scan of `TRANSCRIBE_FOLDERS` entirely. Subgen will still watch for new files if `MONITOR` is enabled, but won't iterate existing files on start. Useful if your library is already subtitled and you only want to catch newly added files. |
| `PLEX_QUEUE_NEXT_EPISODE` | `False` | Auto-queues the *next* Plex episode when Subgen is triggered. |
| `PLEX_QUEUE_SEASON` | `False` | Auto-queues the *entire remaining season* when Subgen is triggered. |
| `PLEX_QUEUE_SERIES` | `False` | Auto-queues the *entire remaining series* when Subgen is triggered. |
| `WEBHOOK_URL_COMPLETED` | `''` | Sends a POST to the `WEBHOOK_URL_COMPLETED` URL with a JSON containing: <br><code>{<br>&nbsp;&nbsp;"event": "transcribed",<br>&nbsp;&nbsp;"file": "/absolute/path/to/video.mkv",<br>&nbsp;&nbsp;"subtitle": "/absolute/path/to/video.en.srt",<br>&nbsp;&nbsp;"language": "en"<br>}</code><br>It will not fire on skips, `/asr` or `/detect-language`. |

### ⏭️ Skip Logic & Audio Targeting
*Prevent Subgen from wasting time on files that don't need subtitles.*

**Directory skip marker:** Place an empty `.subgen_skip` file in any directory to tell Subgen to skip that directory and all of its subdirectories — both during the startup scan and when watching for new files. Useful for completed shows or any section of your library that will never need new subtitles.
```bash
touch "/tv/The Simpsons/.subgen_skip"   # skips all seasons
touch "/tv/Some Show/Season 1/.subgen_skip"  # skips just that season
```

| Variable | Default | Description |
|---|---|---|
| `SKIP_IF_TARGET_SUBTITLES_EXIST` | `True` | Skips if an auto-generated subtitle in your desired language already exists. |
| `SKIP_IF_EXTERNAL_SUBTITLES_EXIST`| `False` | Skips if an external subtitle matching `SUBTITLE_LANGUAGE_NAME` is found. |
| `SKIP_IF_INTERNAL_SUBTITLES_LANGUAGE`| `eng` | Skips if the file contains an embedded sub with this 3-letter code. |
| `SKIP_SUBTITLE_LANGUAGES` | `''` | Pipe-separated list (e.g., `eng\|spa`). Skips if the file *has audio* in these languages. |
| `SKIP_IF_AUDIO_LANGUAGES` | `''` | Pipe-separated list (ISO 639-2). Skips generation if the file has audio tracks in these languages. |
| `PREFERRED_AUDIO_LANGUAGES` | `eng` | Pipe-separated list. If multiple audio tracks exist, prefer transcribing this one. |
| `LIMIT_TO_PREFERRED_AUDIO_LANGUAGE`| `False` | If True, skips files that do not have any audio tracks matching your preferred list. |
| `FORCE_DETECTED_LANGUAGE_TO` | `''` | Force model to this 2-letter language code if it keeps incorrectly detecting audio. |
| `DETECT_LANGUAGE_LENGTH` | `30` | Number of seconds to analyze audio to determine the language. |
| `DETECT_LANGUAGE_OFFSET` | `0` | Number of seconds to skip forward before detecting language (good for avoiding theme songs). |
| `SHOULD_WHISPER_DETECT_AUDIO_LANGUAGE` | `False` | Should Whisper detect language if there is no audio language tagged in the media file. |
| `SKIP_UNKNOWN_LANGUAGE` | `False` | Skip processing if Whisper cannot detect the audio language. |
| `SKIP_ONLY_SUBGEN_SUBTITLES` | `False` | Skips generation only if the file has "subgen" somewhere in the existing subtitle filename. |
| `SKIP_IF_NO_LANGUAGE_BUT_SUBTITLES_EXIST`| `False` | Skips generation if file doesn't have an audio stream marked with a language, but subtitles exist. |
| `IGNORE_FORCED_SUBTITLES` | `True` | When `True`, forced embedded subtitle tracks are excluded from all skip-coverage checks. A file whose only matching subtitle tracks are forced will be treated as having no coverage and transcribed normally. Set to `False` to count forced tracks as full coverage (old behaviour). |

> **Not sure why a file is being skipped?** Use the [Skip Logic Calculator](https://htmlpreview.github.io/?https://github.com/McCloudS/subgen/blob/main/docs/skip-logic-calculator.html) — configure your settings, describe what subtitle files and streams exist, and it shows exactly what triggered the skip.

### 📝 Subtitle Formatting & Preferences
| Variable | Default | Description |
|---|---|---|
| `TRANSCRIBE_OR_TRANSLATE` | `transcribe` | `transcribe` (matches input language) or `translate` (outputs English). |
| `SUBTITLE_LANGUAGE_NAME` | *(detected language)* | Language code to put in the subtitle file name instead of the detected one (e.g. `en`). Many people set `aa` so subgen's file floats to the top of Plex's list. |
| `SUBTITLE_LANGUAGE_NAMING_TYPE`| `ISO_639_2_B` | Format to name files (`ISO_639_1`, `ISO_639_2_T`, `NAME`, `NATIVE`). |
| `LRC_FOR_AUDIO_FILES` | `True` | Generates `.lrc` instead of `.srt` if processing pure audio files (e.g., mp3, flac). |
| `MAX_LINE_LENGTH` | `42` | Maximum characters per subtitle line. Lines are balanced across two rows where possible. |
| `GAP_SPLIT_SECS` | `0.4` | Silence gap in seconds between words that forces a new subtitle card. |
| `APPEND` | `False` | Appends a "Transcribed by whisperAI..." watermark at the very end of the `.srt`. |
| `SHOW_IN_SUBNAME_SUBGEN` | `True` | Adds `.subgen` to the output file name. |
| `SHOW_IN_SUBNAME_MODEL` | `True` | Adds the model used (e.g., `.medium`) to the output file name. |

### 📂 System, Paths & Network Settings
| Variable | Default | Description |
|---|---|---|
| `WEBHOOK_PORT` | `9000` | Port used to listen for webhooks and Bazarr requests. |
| `PUID` / `PGID` | `99` / `100` | Run container as a specific user/group (helps with file permissions). |
| `DEBUG` | `True` | Outputs extra logs, helpful for troubleshooting paths or webhook hits. |
| `RELOAD_SCRIPT_ON_CHANGE` | `False` | (Dev) Auto-reloads uvicorn if `subgen.py` is edited. |
| `UPDATE` | `False` | (Standalone) Will pull the latest `subgen.py` from repo via `launcher.py`. |
| `USE_PATH_MAPPING` | `False` | Set to True if your media server and Subgen map their volumes differently. |
| `PATH_MAPPING_FROM` | `/tv` | Example: The media path on Plex. |
| `PATH_MAPPING_TO` | `/Volumes/TV` | Example: What Subgen natively sees that same path as. |
| `MODEL_PATH` | `./models` | Path where AI models are downloaded and stored. |

### 🎬 Media Server Integration (Metadata Refreshing)
*Required if you want Subgen to automatically generate Subtitles off of Webhook Events from Plex or Jellyfin or to tell Plex or Jellyfin to refresh the show's metadata so the subtitle immediately appears after generation.*
| Variable | Default | Description |
|---|---|---|
| `PLEX_SERVER` | *(None)* | Local Plex address (e.g., `http://192.168.1.100:32400`). |
| `PLEX_TOKEN` | *(None)* | Your Plex Token for API access. |
| `JELLYFIN_SERVER` | *(None)* | Local Jellyfin address (e.g., `http://192.168.1.100:8096`). |
| `JELLYFIN_TOKEN` | *(None)* | Generated API token from Jellyfin UI. |

---

## 🧩 Choosing a model

Set `WHISPER_MODEL` to one of these names. With whisper.cpp, subgen downloads the matching GGML file to `MODEL_PATH` on first use (or set `WHISPER_CPP_MODEL` to a `.bin` file of your own). Memory use is about the file size plus 0.3 to 0.5 GB per concurrent transcription; `large-v3-turbo-q5_0` measured 1.0 GB of video memory.

| `WHISPER_MODEL` | File size | Translate to English? | Notes |
|---|---|---|---|
| `large-v3-turbo` | 1.5 GB | No | Best speed/accuracy balance on a GPU. |
| `large-v3-turbo-q8_0` / `-q5_0` | 833 / 547 MB | No | Quantized turbo: about a third of the memory. `q5_0` is what the benchmarks in this README used. |
| `large-v3` | 2.9 GB | Yes | Most accurate. Use this (or a quantized version) for `TRANSCRIBE_OR_TRANSLATE=translate`. |
| `large-v3-q5_0` | 1.0 GB | Yes | Quantized `large-v3`, good for translation on smaller GPUs. |
| `large-v2`, `large-v2-q8_0`, `large-v2-q5_0`, `large-v1` | 1.0 to 2.9 GB | Yes | Older large models. |
| `distil-large-v3.5`, `distil-large-v3` | 1.4 GB | No | Distilled, English only, fast. `distil-large-v3.fp32` is a 2.9 GB full-precision version. |
| `medium` (default), `medium-q8_0`, `medium-q5_0` | 1.5 GB / 785 MB / 514 MB | Yes | Good on CPU. |
| `small`, `small-q8_0`, `small-q5_1` | 465 / 252 / 181 MB | Yes | Low-power machines. |
| `base`, `base-q8_0`, `base-q5_1` | 141 / 77 / 56 MB | Yes | Low-power hardware; less accurate. |
| `tiny`, `tiny-q8_0`, `tiny-q5_1` | 74 / 41 / 30 MB | Yes | Testing only. |

* Add `.en` for English-only versions of `medium`, `small`, `base` and `tiny` (e.g. `small.en`, `base.en-q5_1`). They are slightly better at English but can't detect or transcribe other languages.
* Quantized names (`-q5_0`, `-q8_0`, `-q5_1`) are whisper.cpp only. faster-whisper accepts the plain names (`large-v3`, `large-v3-turbo`, `distil-large-v3`, `medium`, ...) and sets precision with `COMPUTE_TYPE` instead.
* An unknown name stops subgen with an error that tells you to set `WHISPER_CPP_MODEL`.

---

## 🚀 Speed, memory and concurrent transcriptions

Subgen loads the model once per transcription slot, and `CONCURRENT_TRANSCRIPTIONS` is the number of slots. Extra files wait in the queue; they never start another copy of the model.

* **faster-whisper** loads one model and shares it between the slots.
* **whisper.cpp** runs each transcription as its own `whisper-cli` process with its own copy of the model. Loading takes under a second, but memory grows with every slot. Language detection (used by Bazarr) can briefly add one more copy.

Running two files at once is **not** twice as fast on a GPU: one transcription already keeps the GPU busy. Measured with `large-v3-turbo-q5_0`, 5-minute clips:

| Hardware | Concurrent jobs | Throughput | Memory |
|---|---|---|---|
| Tesla P4 (CUDA) | 1 | 21x real time | 1.0 GB VRAM |
| Tesla P4 (CUDA) | 2 | 24x (+15%) | 2.0 GB VRAM |
| Tesla P4 (CUDA) | 3 | 25x (+19%) | 3.0 GB VRAM |
| 24-core CPU | 1 job, 4 threads | 2.5x | 0.8 GB RAM |
| 24-core CPU | 1 job, 16 threads | 6.0x | 0.8 GB RAM |
| 24-core CPU | 2 jobs, 8 threads each | 6.6x | 1.6 GB RAM |

**Recommendations**
* **GPU:** `CONCURRENT_TRANSCRIPTIONS=1` gives nearly the same throughput with half the memory. Use `2` if you have spare video memory and want a Bazarr request to start while a webhook job is still running (Bazarr requests already jump ahead of queued webhook jobs).
* **CPU:** leave `WHISPER_THREADS` unset. Subgen shares the cores between the slots, which is close to the best you can get.
* **Running out of memory:** lower `CONCURRENT_TRANSCRIPTIONS` first, then use a smaller or quantized model (`large-v3-turbo-q5_0`, `medium`). Memory per slot is roughly the model file size plus a few hundred MB.

---

## 🔄 Upgrading from stable-ts (main)

* **Models are downloaded again on first start.** whisper.cpp uses GGML model files (`ggml-*.bin`), not faster-whisper's. They are derived from `WHISPER_MODEL` and saved to `MODEL_PATH`. Point `WHISPER_CPP_MODEL` at a file to use your own.
* **`CUSTOM_REGROUP` and `WORD_LEVEL_HIGHLIGHT` were removed** with stable-ts. Subgen logs a warning if they are still set. Line length and cue splitting are controlled by `MAX_LINE_LENGTH` and `GAP_SPLIT_SECS`.
* **`TRANSCRIBE_DEVICE` only applies to faster-whisper.** whisper.cpp picks the GPU it can see; use `WHISPER_CPP_DEVICE` to choose one or force `cpu`.
* **Subtitle layout changed:** two-line cues of up to 42 characters per line, at most two sentences per cue, no one-word flashes. Expect fewer, longer cues than stable-ts produced.

---

## 🌎 Supported Audio Languages (via OpenAI)
Afrikaans, Arabic, Armenian, Azerbaijani, Belarusian, Bosnian, Bulgarian, Catalan, Chinese, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, Galician, German, Greek, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian, Japanese, Kannada, Kazakh, Korean, Latvian, Lithuanian, Macedonian, Malay, Marathi, Maori, Nepali, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swahili, Swedish, Tagalog, Tamil, Thai, Turkish, Ukrainian, Urdu, Vietnamese, and Welsh.

---

## 🔗 OpenAI-Compatible API Endpoints

Subgen exposes two endpoints that match the [OpenAI Whisper API](https://platform.openai.com/docs/api-reference/audio), so it can be used as a drop-in backend for any client that targets that API (Open WebUI, Obsidian plugins, etc.).

| Endpoint | Description |
|---|---|
| `POST /v1/audio/transcriptions` | Transcribe audio to text in the source language |
| `POST /v1/audio/translations` | Transcribe and translate audio to English |

**Supported parameters:**

| Parameter | Description |
|---|---|
| `file` | Audio file (any format ffmpeg can decode) |
| `language` | Source language ISO-639-1 code (transcriptions only; auto-detected if omitted) |
| `prompt` | Optional context passed to Whisper |
| `response_format` | `json` (default), `text`, `srt`, `vtt`, `verbose_json` |
| `model` | Accepted but ignored — subgen uses its configured model |
| `temperature` | Accepted but ignored |

**Example:**
```bash
curl -X POST http://<your-ip>:9000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "response_format=json"
# {"text": "..."}
```

`verbose_json` returns segments with start/end timestamps.

**Bazarr / whisper-asr-webservice endpoint (`POST /asr`):** query parameters `task` (`transcribe` or `translate`), `language`, `output` (`srt` default, `vtt`, `txt`, `tsv`, `json`), `encode` (`false` for raw 16 kHz mono PCM, which is what Bazarr sends) and `video_file`. `POST /detect-language` returns `{"detected_language": ..., "language_code": ...}`.

---

## 🪲 Known Issues
* It uses trained AI models; there *will* occasionally be mistranslations or hallucinations based on background noise.

## ❤️ Credits
* [whisper.cpp](https://github.com/ggml-org/whisper.cpp), the transcription engine
* Google & FFmpeg
* [faster-whisper](https://github.com/guillaumekln/faster-whisper)
* [Whisper ASR Webservice](https://github.com/ahmetoner/whisper-asr-webservice) for Bazarr HTTP webhook logic.
* Community Contributors
