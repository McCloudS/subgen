# Subgen

[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate/?hosted_button_id=SU4QQP6LH5PF6)
<img src="https://raw.githubusercontent.com/McCloudS/subgen/main/icon.png" width="200">

<details>
<summary><strong>Updates:</strong></summary>

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

3 Aug 2024: Added SUBGEN_KWARGS environment variable which allows you to override the model.transcribe with most options you'd like from whisper, faster-whisper, or stable-ts. This won't be exposed via the webui, it's best to set directly.

21 Apr 2024: Fixed queuing with thanks to https://github.com/xhzhu0628 @ https://github.com/McCloudS/subgen/pull/85. Bazarr intentionally doesn't follow `CONCURRENT_TRANSCRIPTIONS` because it needs a time sensitive response.

31 Mar 2024: Removed `/subsync` endpoint and general refactoring. Open an issue if you were using it!

24 Mar 2024: ~~Added a 'webui' to configure environment variables. You can use this instead of manually editing the script or using Environment Variables in your OS or Docker (if you want). The config will prioritize OS Env Variables, then the .env file, then the defaults. You can access it at `http://subgen:9000/`~~

23 Mar 2024: Added `CUSTOM_REGROUP` to try to 'clean up' subtitles a bit.  

22 Mar 2024: Added LRC capability via see: `LRC_FOR_AUDIO_FILES | True | Will generate LRC (instead of SRT) files for filetypes: '.mp3', '.flac', '.wav', '.alac', '.ape', '.ogg', '.wma', '.m4a', '.m4b', '.aac', '.aiff'`

21 Mar 2024: Added a 'wizard' into the launcher that will help standalone users get common Bazarr variables configured. See below in Launcher section. Removed 'Transformers' as an option. While I usually don't like to remove features, I don't think anyone is using this and the results are wildly unpredictable and often cause out of memory errors. Added two new environment variables called `USE_MODEL_PROMPT` and `CUSTOM_MODEL_PROMPT`. If `USE_MODEL_PROMPT` is `True` it will use `CUSTOM_MODEL_PROMPT` if set, otherwise will default to using the pre-configured language pairings. These pre-configurated translations are geared towards fixing some audio that may not have punctionation. We can prompt it to try to force the use of punctuation during transcription.

19 Mar 2024: Added a `MONITOR` environment variable. Will 'watch' or 'monitor' your `TRANSCRIBE_FOLDERS` for changes and run on them. Useful if you just want to paste files into a folder and get subtitles.   

6 Mar 2024: Added a `/subsync` endpoint that can attempt to align/synchronize subtitles to a file. Takes audio_file, subtitle_file, language (2 letter code), and outputs an srt.

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

19 Oct 2023: And we're back! Uses faster-whisper and stable-ts. Shouldn't break anything from previous settings, but adds a couple new options that aren't documented at this point in time. As of now, this is not a docker image on dockerhub. The potential intent is to move this eventually to a pure python script, primarily to simplify my efforts. Quick and dirty to meet dependencies: pip or `pip3 install flask requests stable-ts faster-whisper`

This potentially has the ability to use CUDA/Nvidia GPU's, but I don't have one set up yet. Tesla T4 is in the mail!

2 Feb 2023: Added Tautulli webhooks back in. Didn't realize Plex webhooks was PlexPass only. See below for instructions to add it back in.

31 Jan 2023 : Rewrote the script substantially to remove Tautulli and fix some variable handling. For some reason my implementation requires the container to be in host mode. My Plex was giving "401 Unauthorized" when attempt to query from docker subnets during API calls. (**Fixed now, it can be in bridge**)
</details>

---

## 🎬 What is this?
Subgen transcribes your personal media to create subtitles (`.srt` or `.lrc`) from audio/video files. It can transcribe non-English languages to themselves, or translate foreign languages into English. 

It is designed to integrate perfectly with **Bazarr** (as a Whisper Provider), or run via webhooks triggered directly by your **Plex, Emby, Jellyfin, or Tautulli** servers whenever media is added or played. Under the hood, it uses `stable-ts` and `faster-whisper`, fully supporting both CPU and Nvidia GPU (CUDA) transcoding.

## 🤔 Why?
Some shows just won't have subtitles available, or embedded H265 subtitles might be wildly out of sync. This gap-fills everything else by generating highly accurate subtitles locally on your own hardware. 

---

## ⚡ Quick Start: Bazarr (The Bare Minimum)
If you just want to plug Subgen into Bazarr and get going, here is the absolute minimum you need to configure in your Subgen Docker container. **No path mapping or media mounts are needed!**

**1. Set your Environment Variables in Subgen:**
* `TRANSCRIBE_DEVICE`: Set to `cuda` if you have an Nvidia GPU (highly recommended for speed), otherwise leave as `cpu`.
* `WHISPER_MODEL`: Default is `medium`. Try `large-v3-turbo` if you have a GPU with 8GB+ VRAM for faster, highly accurate results.
* `CONCURRENT_TRANSCRIPTIONS`: Default is `2`. Lower to `1` if you are running out of RAM/VRAM.

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

## 🛠 Installation & Setup

### 1. Docker (Recommended)
The easiest way to run Subgen is via Docker. We maintain an image on Docker Hub (`mccloud/subgen`).
* `mccloud/subgen:latest` (Supports both CPU and GPU/CUDA)
* `mccloud/subgen:cpu` (Smaller image, CPU only)

**Crucial Note on Volume Mapping:** If you are using Plex/Emby/Jellyfin/Tautulli webhooks, **Subgen must see your media paths exactly identically to how your media server sees them.** For example, if Plex uses `/Share/media/TV:/tv`, Subgen needs that exact same volume mount. *(Note: This does not apply to Bazarr, which sends audio over HTTP).*

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

*Note: Subgen recently standardized environment variables (e.g., `PLEX_TOKEN`). Legacy names (e.g., `PLEXTOKEN`) are still fully supported for backwards compatibility!*

### 🧠 Core Whisper & AI Settings
| Variable | Default | Description |
|---|---|---|
| `TRANSCRIBE_DEVICE` | `cpu` | Device to transcribe on: `cpu`, `gpu`, or `cuda`. |
| `WHISPER_MODEL` | `medium` | Model to use: `tiny`, `base`, `small`, `medium`, `large-v3`, `distil-large-v3`, `large-v3-turbo`, etc. |
| `CONCURRENT_TRANSCRIPTIONS` | `2` | Number of files to process in parallel. |
| `WHISPER_THREADS` | `4` | Number of CPU threads to use during computation. |
| `COMPUTE_TYPE` | `auto` | Precision quantization mapping (e.g., `float16`, `int8`). See [CTranslate2 docs](https://github.com/OpenNMT/CTranslate2/blob/master/docs/quantization.md). |
| `CLEAR_VRAM_ON_COMPLETE` | `True` | Do garbage collection and clear the model from VRAM when the queue is empty. |
| `MODEL_CLEANUP_DELAY` | `30` | Seconds to wait before clearing the Whisper model from memory. |
| `ASR_TIMEOUT` | `18000` | Seconds to wait before timing out a transcription request (default 5 hours). |
| `SUBGEN_KWARGS` | `{}` | JSON dict to pass pure kwargs to Whisper (e.g. `{'vad': True}`). For advanced users. |

### ⚡ Processing Triggers & Queuing
*(Not relevant for Bazarr users)*
| Variable | Default | Description |
|---|---|---|
| `PROCESS_ADDED_MEDIA` | `True` | Generate subs for newly added media (when triggered by webhook). |
| `PROCESS_MEDIA_ON_PLAY` | `True` | Generate subs for media when it is played (when triggered by webhook). |
| `TRANSCRIBE_FOLDERS` | `''` | Pipe-separated list (e.g., `/tv&#124;/movies`) to recurse through and queue existing media. |
| `MONITOR` | `False` | Actively watches `TRANSCRIBE_FOLDERS` in real-time for newly pasted files. |
| `PLEX_QUEUE_NEXT_EPISODE` | `False` | Auto-queues the *next* Plex episode when Subgen is triggered. |
| `PLEX_QUEUE_SEASON` | `False` | Auto-queues the *entire remaining season* when Subgen is triggered. |
| `PLEX_QUEUE_SERIES` | `False` | Auto-queues the *entire remaining series* when Subgen is triggered. |
| `WEBHOOK_URL_COMPLETED` | `''` | Sends a POST to the `WEBHOOK_URL_COMPLETED` URL with a JSON containing: <br><code>{<br>&nbsp;&nbsp;"event": "transcribed",<br>&nbsp;&nbsp;"file": "/absolute/path/to/video.mkv",<br>&nbsp;&nbsp;"subtitle": "/absolute/path/to/video.en.srt",<br>&nbsp;&nbsp;"language": "en"<br>}</code><br>It will not fire on skips, `/asr` or `/detect-language`. |

### ⏭️ Skip Logic & Audio Targeting
*Prevent Subgen from wasting time on files that don't need subtitles.*
| Variable | Default | Description |
|---|---|---|
| `SKIP_IF_TARGET_SUBTITLES_EXIST` | `True` | Skips if an auto-generated subtitle in your desired language already exists. |
| `SKIP_IF_EXTERNAL_SUBTITLES_EXIST`| `False` | Skips if an external subtitle matching `SUBTITLE_LANGUAGE_NAME` is found. |
| `SKIP_IF_INTERNAL_SUBTITLES_LANGUAGE`| `eng` | Skips if the file contains an embedded sub with this 3-letter code. |
| `SKIP_SUBTITLE_LANGUAGES` | `''` | Pipe-separated list (e.g., `eng&#124;spa`). Skips if the file *has audio* in these languages. |
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

### 📝 Subtitle Formatting & Preferences
| Variable | Default | Description |
|---|---|---|
| `TRANSCRIBE_OR_TRANSLATE` | `transcribe` | `transcribe` (matches input language) or `translate` (outputs English). |
| `SUBTITLE_LANGUAGE_NAME` | `aa` | Subtitle file name language code (e.g. `en`). Defaults to `aa` so it floats to the top of Plex's list. |
| `SUBTITLE_LANGUAGE_NAMING_TYPE`| `ISO_639_2_B` | Format to name files (`ISO_639_1`, `ISO_639_2_T`, `NAME`, `NATIVE`). |
| `LRC_FOR_AUDIO_FILES` | `True` | Generates `.lrc` instead of `.srt` if processing pure audio files (e.g., mp3, flac). |
| `WORD_LEVEL_HIGHLIGHT` | `False` | Highlights words dynamically as they are spoken in the subtitle. |
| `APPEND` | `False` | Appends a "Transcribed by whisperAI..." watermark at the very end of the `.srt`. |
| `SHOW_IN_SUBNAME_SUBGEN` | `True` | Adds `.subgen` to the output file name. |
| `SHOW_IN_SUBNAME_MODEL` | `True` | Adds the model used (e.g., `.medium`) to the output file name. |
| `CUSTOM_REGROUP` | `cm_sl=84_sl=42++++++1` | Stable-TS grouping. Try to 'clean up' subtitles a bit. Set to `default` to use base Stable-TS. |

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

## 🌎 Supported Audio Languages (via OpenAI)
Afrikaans, Arabic, Armenian, Azerbaijani, Belarusian, Bosnian, Bulgarian, Catalan, Chinese, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, Galician, German, Greek, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian, Japanese, Kannada, Kazakh, Korean, Latvian, Lithuanian, Macedonian, Malay, Marathi, Maori, Nepali, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swahili, Swedish, Tagalog, Tamil, Thai, Turkish, Ukrainian, Urdu, Vietnamese, and Welsh.

---

## 🪲 Known Issues
* It uses trained AI models; there *will* occasionally be mistranslations or hallucinations based on background noise.

## ❤️ Credits
* [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) for original implementation
* Google & FFmpeg
* [stable-ts](https://github.com/jianfch/stable-ts) & [faster-whisper](https://github.com/guillaumekln/faster-whisper)
* [Whisper ASR Webservice](https://github.com/ahmetoner/whisper-asr-webservice) for Bazarr HTTP webhook logic.
* Community Contributors
