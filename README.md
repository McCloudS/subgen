[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate/?hosted_button_id=SU4QQP6LH5PF6)
<img src="https://raw.githubusercontent.com/McCloudS/subgen/main/icon.png" width="200">

<details>
<summary>Updates:</summary>

Feb 2026: Contributor helped cut the GPU container size in half.  Added `ASR_TIMEOUT` as environment variable to timeout ASR endpoint transcriptions after X seconds.

31 Jan 2026: Added the ability to run the container 'rootless', accepts `PUID` and `PGID` as environment variables and _should_ take `user` at a container level (Podman), let me know!.

13 Jan 2026: Probably fixed the runaway memory problems for CPU only.  Added `MODEL_CLEANUP_DELAY` which will wait X seconds before purging the model to clear up (V)RAM.  This mostly helps with Bazarr or when concurrent transcriptions is 1.  Rewrote ASR (Bazarr) queuing so it should respect queing and follow concurrent transcriptions.  Also fixed the error when too many Bazarr or ASR requests would start to fail.  

26 Aug 2025: Renamed environment variables to make them slightly easier to understand.  Currently maintains backwards compatibility. See https://github.com/McCloudS/subgen/pull/229

12 Aug 2025: Added distil-large-v3.5

7 Feb: Fixed (V)RAM clearing, added PLEX_QUEUE_SEASON, other extraneous fixes or refactorting.  

23 Dec: Added PLEX_QUEUE_NEXT_EPISODE and PLEX_QUEUE_SERIES.  Will automatically start generating subtitles for the next episode in your series, or queue the whole series.  

4 Dec: Added more ENV settings: DETECT_LANGUAGE_OFFSET, PREFERRED_AUDIO_LANGUAGES, SKIP_IF_AUDIO_TRACK_IS, ONLY_SKIP_IF_SUBGEN_SUBTITLE, SKIP_UNKNOWN_LANGUAGE, SKIP_IF_LANGUAGE_IS_NOT_SET_BUT_SUBTITLES_EXIST, SHOULD_WHISPER_DETECT_AUDIO_LANGUAGE

30 Nov 2024: Signifcant refactoring and handling by Muisje.  Added language code class for more robustness and flexibility and ability to separate audio tracks to make sure you get the one you want.  New ENV Variables: SUBTITLE_LANGUAGE_NAMING_TYPE, SKIP_IF_AUDIO_TRACK_IS, PREFERRED_AUDIO_LANGUAGE, SKIP_IF_TO_TRANSCRIBE_SUB_ALREADY_EXIST

    There will be some minor hiccups, so please identify them as we work through this major overhaul.

22 Nov 2024: Updated to support large-v3-turbo

30 Sept 2024: Removed webui

5 Sept 2024: Fixed Emby response to a test message/notification.  Clarified Emby/Plex/Jellyfin instructions for paths.

14 Aug 2024: Cleaned up usage of kwargs across the board a bit.  Added ability for /asr to encode or not, so you don't need to worry about what files/formats you upload.

3 Aug 2024: Added SUBGEN_KWARGS environment variable which allows you to override the model.transcribe with most options you'd like from whisper, faster-whisper, or stable-ts.  This won't be exposed via the webui, it's best to set directly.

21 Apr 2024: Fixed queuing with thanks to https://github.com/xhzhu0628 @ https://github.com/McCloudS/subgen/pull/85.  Bazarr intentionally doesn't follow `CONCURRENT_TRANSCRIPTIONS` because it needs a time sensitive response.

31 Mar 2024: Removed `/subsync` endpoint and general refactoring.  Open an issue if you were using it!

24 Mar 2024: ~~Added a 'webui' to configure environment variables.  You can use this instead of manually editing the script or using Environment Variables in your OS or Docker (if you want).  The config will prioritize OS Env Variables, then the .env file, then the defaults.  You can access it at `http://subgen:9000/`~~

23 Mar 2024: Added `CUSTOM_REGROUP` to try to 'clean up' subtitles a bit.  

22 Mar 2024: Added LRC capability via see: `'LRC_FOR_AUDIO_FILES' | True | Will generate LRC (instead of SRT) files for filetypes: '.mp3', '.flac', '.wav', '.alac', '.ape', '.ogg', '.wma', '.m4a', '.m4b', '.aac', '.aiff' |`

21 Mar 2024: Added a 'wizard' into the launcher that will help standalone users get common Bazarr variables configured.  See below in Launcher section.  Removed 'Transformers' as an option.  While I usually don't like to remove features, I don't think anyone is using this and the results are wildly unpredictable and often cause out of memory errors.  Added two new environment variables called `USE_MODEL_PROMPT` and `CUSTOM_MODEL_PROMPT`.  If `USE_MODEL_PROMPT` is `True` it will use `CUSTOM_MODEL_PROMPT` if set, otherwise will default to using the pre-configured language pairings, such as: `"en": "Hello, welcome to my lecture.",
    "zh": "你好，欢迎来到我的讲座。"`  These pre-configurated translations are geared towards fixing some audio that may not have punctionation.  We can prompt it to try to force the use of punctuation during transcription.

19 Mar 2024: Added a `MONITOR` environment variable.  Will 'watch' or 'monitor' your `TRANSCRIBE_FOLDERS` for changes and run on them.  Useful if you just want to paste files into a folder and get subtitles.   

6 Mar 2024: Added a `/subsync` endpoint that can attempt to align/synchronize subtitles to a file.  Takes audio_file, subtitle_file, language (2 letter code), and outputs an srt.

5 Mar 2024: Cleaned up logging. Added timestamps option (if Debug = True, timestamps will print in logs).

4 Mar 2024: Updated Dockerfile CUDA to 12.2.2 (From CTranslate2).  Added endpoint `/status` to return Subgen version.  Can also use distil models now!  See variables below!

29 Feb 2024: Changed sefault port to align with whisper-asr and deconflict other consumers of the previous port.

11 Feb 2024: Added a 'launcher.py' file for Docker to prevent huge image downloads. Now set UPDATE to True if you want pull the latest version, otherwise it will default to what was in the image on build.  Docker builds will still be auto-built on any commit.  If you don't want to use the auto-update function, no action is needed on your part and continue to update docker images as before.  Fixed bug where detect-langauge could return an empty result.  Reduced useless debug output that was spamming logs and defaulted DEBUG to True.  Added APPEND, which will add f"Transcribed by whisperAI with faster-whisper ({whisper_model}) on {datetime.now()}" at the end of a subtitle.

10 Feb 2024: Added some features from JaiZed's branch such as skipping if SDH subtitles are detected, functions updated to also be able to transcribe audio files, allow individual files to be manually transcribed, and a better implementation of forceLanguage. Added `/batch` endpoint (Thanks JaiZed).  Allows you to navigate in a browser to http://subgen_ip:9000/docs and call the batch endpoint which can take a file or a folder to manually transcribe files.  Added CLEAR_VRAM_ON_COMPLETE, HF_TRANSFORMERS, HF_BATCH_SIZE.  Hugging Face Transformers boast '9x increase', but my limited testing shows it's comparable to faster-whisper or slightly slower.  I also have an older 8gb GPU.  Simplest way to persist HF Transformer models is to set "HF_HUB_CACHE" and set it to "/subgen/models" for Docker (assuming you have the matching volume).

8 Feb 2024: Added FORCE_DETECTED_LANGUAGE_TO to force a wrongly detected language.  Fixed asr to actually use the language passed to it.  

5 Feb 2024: General housekeeping, minor tweaks on the TRANSCRIBE_FOLDERS function.

28 Jan 2024: Fixed issue with ffmpeg python module not importing correctly.  Removed separate GPU/CPU containers.  Also removed the script from installing packages, which should help with odd updates I can't control (from other packages/modules). The image is a couple gigabytes larger, but allows easier maintenance.  

19 Dec 2023: Added the ability for Plex and Jellyfin to automatically update metadata so the subtitles shows up properly on playback. (See https://github.com/McCloudS/subgen/pull/33 from Rikiar73574)  

31 Oct 2023: Added Bazarr support via Whipser provider.

25 Oct 2023: Added Emby (IE http://192.168.1.111:9000/emby) support and TRANSCRIBE_FOLDERS, which will recurse through the provided folders and generate subtitles.  It's geared towards attempting to transcribe existing media without using a webhook.

23 Oct 2023: There are now two docker images, ones for CPU (it's smaller): mccloud/subgen:latest, mccloud/subgen:cpu, the other is for cuda/GPU: mccloud/subgen:cuda.  I also added Jellyfin support and considerable cleanup in the script. I also renamed the webhooks, so they will require new configuration/updates on your end. Instead of /webhook they are now /plex, /tautulli, and /jellyfin.

22 Oct 2023: The script should have backwards compability with previous envirionment settings, but just to be sure, look at the new options below.  If you don't want to manually edit your environment variables, just edit the script manually. While I have added GPU support, I haven't tested it yet.

19 Oct 2023: And we're back!  Uses faster-whisper and stable-ts.  Shouldn't break anything from previous settings, but adds a couple new options that aren't documented at this point in time.  As of now, this is not a docker image on dockerhub.  The potential intent is to move this eventually to a pure python script, primarily to simplify my efforts.  Quick and dirty to meet dependencies: pip or `pip3 install flask requests stable-ts faster-whisper`

This potentially has the ability to use CUDA/Nvidia GPU's, but I don't have one set up yet.  Tesla T4 is in the mail!

2 Feb 2023: Added Tautulli webhooks back in.  Didn't realize Plex webhooks was PlexPass only.  See below for instructions to add it back in.

31 Jan 2023 : Rewrote the script substantially to remove Tautulli and fix some variable handling.  For some reason my implementation requires the container to be in host mode.  My Plex was giving "401 Unauthorized" when attempt to query from docker subnets during API calls. (**Fixed now, it can be in bridge**)

</details>

# What is this?

This will transcribe your personal media on a Plex, Emby, or Jellyfin server to create subtitles (.srt) from audio/video files with the following languages: https://github.com/McCloudS/subgen#audio-languages-supported-via-openai and transcribe or translate them into english. It can also be used as a Whisper provider in Bazarr (See below instructions). It technically has support to transcribe from a foreign langauge to itself (IE Japanese > Japanese, see [TRANSCRIBE_OR_TRANSLATE](https://github.com/McCloudS/subgen#variables)). It is currently reliant on webhooks from Jellyfin, Emby, Plex, or Tautulli. This uses stable-ts and faster-whisper which can use both Nvidia GPUs and CPUs.

# Why?

Honestly, I built this for me, but saw the utility in other people maybe using it.  This works well for my use case.  Since having children, I'm either deaf or wanting to have everything quiet.  We watch EVERYTHING with subtitles now, and I feel like I can't even understand the show without them.  I use Bazarr to auto-download, and gap fill with Plex's built-in capability.  This is for everything else.  Some shows just won't have subtitles available for some reason or another, or in some cases on my H265 media, they are wildly out of sync. 

# What can it do?

* Create .srt subtitles when a media file is added or played which triggers off of Jellyfin, Plex, or Tautulli webhooks. It can also be called via the Whisper provider inside Bazarr.

# How do I set it up?

## Install/Setup

### Standalone/Without Docker

Install python3 (Whisper supports Python 3.9-3.11), ffmpeg, and download launcher.py from this repository.  Then run it: `python3 launcher.py -u -i -s`. You need to have matching paths relative to your Plex server/folders, or use USE_PATH_MAPPING.  Paths are not needed if you are only using Bazarr. You will need the appropriate NVIDIA drivers installed minimum of CUDA Toolkit 12.3 (12.3.2 is known working): https://developer.nvidia.com/cuda-toolkit-archive

Note: If you have previously had Subgen running in standalone, you may need to run `pip install --upgrade --force-reinstall faster-whisper git+https://github.com/jianfch/stable-ts.git` to force the install of the newer stable-ts package.

#### Using Launcher

launcher.py can launch subgen for you and automate the setup and can take the following options:
![image](https://github.com/McCloudS/subgen/assets/64094529/081f95b2-7a09-498f-a39e-5ea66e0bc7e1)

Using `-s` for Bazarr setup:
![image](https://github.com/McCloudS/subgen/assets/64094529/ade1b886-3b99-4f80-95ac-bb28608259bb)



### Docker

The dockerfile is in the repo along with an example docker-compose file, and is also posted on dockerhub (mccloud/subgen). 

If using Subgen without Bazarr, you MUST mount your media volumes in subgen the same way Plex (or your media server) sees them.  For example, if Plex uses "/Share/media/TV:/tv" you must have that identical volume in subgen.  

`"${APPDATA}/subgen/models:/subgen/models"` is just for storage of the language models.  This isn't necessary, but you will have to redownload the models on any new image pulls if you don't use it.  

`"${APPDATA}/subgen/subgen.py:/subgen/subgen.py"` If you want to control the version of subgen.py by yourself.  Launcher.py can still be used to download a newer version.

If you want to use a GPU, you need to map it accordingly.  

#### Unraid

While Unraid doesn't have an app or template for quick install, with minor manual work, you can install it.  See [https://github.com/McCloudS/subgen/discussions/137](https://github.com/McCloudS/subgen/discussions/137) for pictures and steps.

## Bazarr

You only need to confiure the Whisper Provider as shown below: <br>
![bazarr_configuration](https://wiki.bazarr.media/Additional-Configuration/images/whisper_config.png) <br>
The Docker Endpoint is the ip address and port of your subgen container (IE http://192.168.1.111:9000) See https://wiki.bazarr.media/Additional-Configuration/Whisper-Provider/ for more info.  **127.0.0.1 WILL NOT WORK IF YOU ARE RUNNING BAZARR IN A DOCKER CONTAINER!** I recomend not enabling using the Bazarr provider with other webhooks in Subgen, or you will likely be generating duplicate subtitles. If you are using Bazarr, path mapping isn't necessary, as Bazarr sends the file over http.

**The defaults of Subgen will allow it to run in Bazarr with zero configuration.  However, you will probably want to change, at a minimum, `TRANSCRIBE_DEVICE` and `WHISPER_MODEL`.**

## Plex

Create a webhook in Plex that will call back to your subgen address, IE: http://192.168.1.111:9000/plex see: https://support.plex.tv/articles/115002267687-webhooks/  You will also need to generate the token to use it.  Remember, Plex and Subgen need to be able to see the exact same files at the exact same paths, otherwise you need `USE_PATH_MAPPING`.

## Emby

All you need to do is create a webhook in Emby pointing to your subgen IE: `http://192.168.154:9000/emby`, set `Request content type` to `multipart/form-data` and configure your desired events (Usually, `New Media Added`, `Start`, and `Unpause`).  See https://github.com/McCloudS/subgen/discussions/115#discussioncomment-10569277 for screenshot examples.

Emby was really nice and provides good information in their responses, so we don't need to add an API token or server url to query for more information.

Remember, Emby and Subgen need to be able to see the exact same files at the exact same paths, otherwise you need `USE_PATH_MAPPING`.

## Tautulli

Create the webhooks in Tautulli with the following settings:
Webhook URL: http://yourdockerip:9000/tautulli
Webhook Method: Post
Triggers: Whatever you want, but you'll likely want "Playback Start" and "Recently Added"
Data: Under Playback Start, JSON Header will be:
```json 
{ "source":"Tautulli" }
```
Data:
```json
{
            "event":"played",
            "file":"{file}",
            "filename":"{filename}",
            "mediatype":"{media_type}"
}
```
Similarly, under Recently Added, Header is: 
```json
{ "source":"Tautulli" }
```
Data:
```json
{
            "event":"added",
            "file":"{file}",
            "filename":"{filename}",
            "mediatype":"{media_type}"
}
```
## Jellyfin

First, you need to install the Jellyfin webhooks plugin.  Then you need to click "Add Generic Destination", name it anything you want, webhook url is your subgen info (IE http://192.168.1.154:9000/jellyfin).  Next, check Item Added, Playback Start, and Send All Properties.  Last, "Add Request Header" and add the Key: `Content-Type` Value: `application/json`<br><br>Click Save and you should be all set!

Remember, Jellyfin and Subgen need to be able to see the exact same files at the exact same paths, otherwise you need `USE_PATH_MAPPING`.

## Variables

You can define the port via environment variables, but the endpoints are static.

The following environment variables are available in Docker.  They will default to the values listed below.
| Variable                  | Default Value          | Description                                                                                                                                                                                                                                                                               |
|---------------------------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| TRANSCRIBE_DEVICE         | 'cpu'                  | Can transcribe via gpu (Cuda only) or cpu.  Takes option of "cpu", "gpu", "cuda".                                                                                                                     |
| WHISPER_MODEL             | 'medium'               | Can be:'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1','large-v2', 'large-v3', 'large', 'distil-large-v2', 'distil-large-v3', 'distil-large-v3.5', 'distil-medium.en', 'distil-small.en', 'large-v3-turbo'                                   |
| CONCURRENT_TRANSCRIPTIONS | 2                      | Number of files it will transcribe in parallel                                                                                                                                                                                                                                            |
| WHISPER_THREADS           | 4                      | number of threads to use during computation                                                                                                                                                                                                                                               |
| MODEL_PATH                | './models'                    | This is where the WHISPER_MODEL will be stored.  This defaults to placing it where you execute the script in the folder 'models'                                                                                                                                                                              |
| PROCESS_ADDED_MEDIA            | True                   | will gen subtitles for all media added regardless of existing external/embedded subtitles (based off of SKIP_IF_INTERNAL_SUBTITLES_LANGUAGE)                                                                                                                                                            |
| PROCESS_MEDIA_ON_PLAY           | True                   | will gen subtitles for all played media regardless of existing external/embedded subtitles (based off of SKIP_IF_INTERNAL_SUBTITLES_LANGUAGE)                                                                                                                                                           |
| SUBTITLE_LANGUAGE_NAME               | 'aa'                   | allows you to pick what it will name the subtitle. Instead of using EN, I'm using AA, so it doesn't mix with exiting external EN subs, and AA will populate higher on the list in Plex. This will override the Whisper detected language for a file name.                                                                                                   |
| SKIP_IF_INTERNAL_SUBTITLES_LANGUAGE     | 'eng'                  | Will not generate a subtitle if the file has an internal sub matching the 3 letter code of this variable (See https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)                                                                                                                      |
| WORD_LEVEL_HIGHLIGHT      | False                  | Highlights each words as it's spoken in the subtitle.  See example video @ https://github.com/jianfch/stable-ts                                                                                                                                                                           |
| PLEX_SERVER                | 'http://plex:32400'    | This needs to be set to your local plex server address/port                                                                                                                                                                                                                               |
| PLEX_TOKEN                 | 'token here'           | This needs to be set to your plex token found by https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/                                                                                                                                                 |
| JELLYFIN_SERVER            | 'http://jellyfin:8096' | Set to your Jellyfin server address/port                                                                                                                                                                                                                                                  |
| JELLYFIN_TOKEN             | 'token here'           | Generate a token inside the Jellyfin interface                                                                                                                                                                                                                                            |
| WEBHOOK_PORT               | 9000                   | Change this if you need a different port for your webhook                                                                                                                                                                                                                                 |
| USE_PATH_MAPPING          | False                  | Similar to sonarr and radarr path mapping, this will attempt to replace paths on file systems that don't have identical paths.  Currently only support for one path replacement. Examples below.                                                                                          |
| PATH_MAPPING_FROM         | '/tv'                  | This is the path of my media relative to my Plex server                                                                                                                                                                                                                                   |
| PATH_MAPPING_TO           | '/Volumes/TV'          | This is the path of that same folder relative to my Mac Mini that will run the script                                                                                                                                                                                                     |
| TRANSCRIBE_FOLDERS        | ''                     | Takes a pipe '\|' separated list (For example: /tv\|/movies\|/familyvideos) and iterates through and adds those files to be queued for subtitle generation if they don't have internal subtitles                                                                                              |
| TRANSCRIBE_OR_TRANSLATE   | 'transcribe'            | Takes either 'transcribe' or 'translate'.  Transcribe will transcribe the audio in the same language as the input. Translate will transcribe and translate into English. | 
| COMPUTE_TYPE | 'auto' | Set compute-type using the following information: https://github.com/OpenNMT/CTranslate2/blob/master/docs/quantization.md |
| DEBUG                     | True                  | Provides some debug data that can be helpful to troubleshoot path mapping and other issues. Fun fact, if this is set to true, any modifications to the script will auto-reload it (if it isn't actively transcoding).  Useful to make small tweaks without re-downloading the whole file. |
| FORCE_DETECTED_LANGUAGE_TO | '' | This is to force the model to a language instead of the detected one, takes a 2 letter language code.  For example, your audio is French but keeps detecting as English, you would set it to 'fr' |
| CLEAR_VRAM_ON_COMPLETE | True | This will delete the model and do garbage collection when queue is empty.  Good if you need to use the VRAM for something else. |
| UPDATE | False | Will pull latest subgen.py from the repository if True.  False will use the original subgen.py built into the Docker image.  Standalone users can use this with launcher.py to get updates. |
| APPEND | False | Will add the following at the end of a subtitle: "Transcribed by whisperAI with faster-whisper ({whisper_model}) on {datetime.now()}"
| MONITOR | False | Will monitor `TRANSCRIBE_FOLDERS` for real-time changes to see if we need to generate subtitles |
| USE_MODEL_PROMPT | False | When set to `True`, will use the default prompt stored in greetings_translations "Hello, welcome to my lecture." to try and force the use of punctuation in transcriptions that don't. Automatic `CUSTOM_MODEL_PROMPT` will only work with ASR, but can still be set manually like so: `USE_MODEL_PROMPT=True and CUSTOM_MODEL_PROMPT=Hello, welcome to my lecture.`  |
| CUSTOM_MODEL_PROMPT | '' | If `USE_MODEL_PROMPT` is `True`, you can override the default prompt (See: https://medium.com/axinc-ai/prompt-engineering-in-whisper-6bb18003562d for great examples). |
| LRC_FOR_AUDIO_FILES | True | Will generate LRC (instead of SRT) files for filetypes: '.mp3', '.flac', '.wav', '.alac', '.ape', '.ogg', '.wma', '.m4a', '.m4b', '.aac', '.aiff' | 
| CUSTOM_REGROUP | 'cm_sl=84_sl=42++++++1' | Attempts to regroup some of the segments to make a cleaner looking subtitle.  Setting to `default` will use default Stable-TS settings. See https://github.com/McCloudS/subgen/issues/68 for discussion. |
| DETECT_LANGUAGE_LENGTH | 30 | Detect language on the first x seconds of the audio. |
| SKIP_IF_EXTERNAL_SUBTITLES_EXIST | False | Skip subtitle generation if an external subtitle with the same language code as NAMESUBLANG is present. Used for the case of not regenerating subtitles if I already have `Movie (2002).NAMESUBLANG.srt` from a non-subgen source. |
| SUBGEN_KWARGS | '{}' | Takes a kwargs python dictionary of options you would like to add/override.  For advanced users.  An example would be `{'vad': True, 'prompt_reset_on_temperature': 0.35}` |
| SKIP_SUBTITLE_LANGUAGES | '' | Takes a pipe separated `\|` list of 3 letter language codes to not generate subtitles for example 'eng\|deu'|
| SUBTITLE_LANGUAGE_NAMING_TYPE | 'ISO_639_2_B' | The type of naming format desired, such as 'ISO_639_1', 'ISO_639_2_T', 'ISO_639_2_B', 'NAME', or 'NATIVE', for example: ("es", "spa", "spa", "Spanish", "Español") |
| SKIP_SUBTITLE_LANGUAGES | '' | Takes a pipe separated `\|` list of 3 letter language codes to skip if the file has audio in that language.  This could be used to skip generating subtitles for a language you don't want, like, I speak English, don't generate English subtitles (for example: 'eng\|deu')|
| PREFERRED_AUDIO_LANGUAGE | 'eng' | If there are multiple audio tracks in a file, it will prefer this setting |
| SKIP_IF_TARGET_SUBTITLES_EXIST | True | Skips generation of subtitle if a file matches our desired language already. |
| DETECT_LANGUAGE_OFFSET | 0 | Allows you to shift when to run detect_language, geared towards avoiding introductions or songs. |
| PREFERRED_AUDIO_LANGUAGES | 'eng' | Pipe separated list |
| SKIP_IF_AUDIO_TRACK_IS | '' | Takes a pipe separated list of ISO 639-2 languages. Skips generation of subtitle if the file has the audio file listed. |
| SKIP_ONLY_SUBGEN_SUBTITLES | False | Skips generation of subtitles if the file has "subgen" somewhere in the same |
| SKIP_UNKNOWN_LANGUAGE | False | Skips generation if the file has an unknown language |
| SKIP_IF_NO_LANGUAGE_BUT_SUBTITLES_EXIST | False | Skips generation if file doesn't have an audio stream marked with a language |
| SHOULD_WHISPER_DETECT_AUDIO_LANGUAGE | False | Should Whisper try to detect the language if there is no audio language specified via force langauge |
| PLEX_QUEUE_NEXT_EPISODE | False | Will queue the next Plex series episode for subtitle generation if subgen is triggered. |
| PLEX_QUEUE_SEASON | False | Will queue the rest of the Plex season for subtitle generation if subgen is triggered. |
| PLEX_QUEUE_SERIES | False | Will queue the whole Plex series for subtitle generation if subgen is triggered. |
| SHOW_IN_SUBNAME_SUBGEN | True | Adds subgen to the subtitle file name. |
| SHOW_IN_SUBNAME_MODEL | True | Adds Whisper model name to the subtitle file name. |
| MODEL_CLEANUP_DELAY | 30 | Seconds to wait before clearing the Whisper model from memory. |
| PUID | 99 | UserID (see: https://docs.linuxserver.io/images/docker-bazarr/#user-group-identifiers) |
| PGID | 100 | GroupID (see: https://docs.linuxserver.io/images/docker-bazarr/#user-group-identifiers) |
| ASR_TIMEOUT | 18000 | In seconds (defaults to 5 hours), the amount of time to wait before killing an ASR task for not finishing. |

### Images:
`mccloud/subgen:latest` is GPU or CPU <br>
`mccloud/subgen:cpu` is for CPU only (slightly smaller image)
<br><br>

# What are the limitations/problems?

* I made it and know nothing about formal deployment for python coding.  
* It's using trained AI models to transcribe, so it WILL mess up

# What's next?  

Fix documentation and make it prettier!
  
# Audio Languages Supported (via OpenAI)

Afrikaans, Arabic, Armenian, Azerbaijani, Belarusian, Bosnian, Bulgarian, Catalan, Chinese, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, Galician, German, Greek, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian, Japanese, Kannada, Kazakh, Korean, Latvian, Lithuanian, Macedonian, Malay, Marathi, Maori, Nepali, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swahili, Swedish, Tagalog, Tamil, Thai, Turkish, Ukrainian, Urdu, Vietnamese, and Welsh.

# Known Issues

At this time, if you have high CPU usage when not actively transcribing on the CPU only docker, try the GPU one.

# Additional reading:

* https://github.com/openai/whisper (Original OpenAI project)
* https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes (2 letter subtitle codes)

# Credits:  
* Whisper.cpp (https://github.com/ggerganov/whisper.cpp) for original implementation
* Google
* ffmpeg
* https://github.com/jianfch/stable-ts
* https://github.com/guillaumekln/faster-whisper
* Whipser ASR Webservice (https://github.com/ahmetoner/whisper-asr-webservice) for how to implement Bazarr webhooks.
