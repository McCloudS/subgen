[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate/?hosted_button_id=SU4QQP6LH5PF6)

Updates:

19 Oct 2023: And we're back!  Uses faster-whisper and stable-ts.  Shouldn't break anything from previous settings, but adds a couple new options that aren't documented at this point in time.  As of now, this is not a docker image on dockerhub.  The potential intent is to move this eventually to a pure python script, primarily to simplify my efforts.  Quick and dirty to meet dependencies: pip or `pip3 install flask requests stable-ts faster-whisper`

This potentially has the ability to use CUDA/Nvidia GPU's, but I don't have one set up yet.  Tesla T4 is in the mail!

2 Feb 2023: Added Tautulli webhooks back in.  Didn't realize Plex webhooks was PlexPass only.  See below for instructions to add it back in.

31 Jan 2023 : Rewrote the script substantially to remove Tautulli and fix some variable handling.  For some reason my implementation requires the container to be in host mode.  My Plex was giving "401 Unauthorized" when attempt to query from docker subnets during API calls.

Howdy all,

This is a project I've had running for a bit, then cleaned up for 'release' while the kids were sleeping.  It's more of a POC, piece of crap, or a proof of concept.  This was also my first ever Python usage.

# BLUF:  Someone else use this idea (not the code!) as a jumping off point.

# What is this?

This is a half-assed attempt of transcribing subtitles (.srt) from your personal media in a Plex server using a CPU.  It is currently reliant on webhooks from Plex.  Why? During my limited testing, Plex was VERY sporadically actually sending out their webhooks using their built-in functionality (https://support.plex.tv/articles/115002267687-webhooks).  This uses whisper.cpp which is an implementation of OpenAI's Whisper model to use CPUs (Do your own research!).  While CPUs obviously aren't super efficient at this, but my server sits idle 99% of the time, so this worked great for me.  

# Why?

Honestly, I built this for me, but saw the utility in other people maybe using it.  This works well for my use case.  Since having children, I'm either deaf or wanting to have everything quiet.  We watch EVERYTHING with subtitles now, and I feel like I can't even understand the show without them.  I use Bazarr to auto-download, and gap fill with Plex's built-in capability.  This is for everything else.  Some shows just won't have subtitles available for some reason or another, or in some cases on my H265 media, they are wildly out of sync. 

# What can it do?

* Create .srt subtitles when a SINGLE media file is added or played via Plex which triggers off of Tautulli webhooks.  

# How do I set it up?

Use the example docker-compose or build your own, just make sure you define PLEXTOKEN and PLEXSERVER at a minimum.  Can it be run without Docker?  Mostly.  [See below](https://github.com/McCloudS/subgen/blob/main/README.md#running-without-docker)

You can now pull the image directly from Dockerhub:
```
docker pull mccloud/subgen
```

## Plex

Create a webhook in Plex that will call back to your subgen address, IE: 192.168.1.111:8090/webhook see: https://support.plex.tv/articles/115002267687-webhooks/

## Tautulli

Create the webhooks in Tautulli with the following settings:
Webhook URL: http://yourdockerip:8090/webhook
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

## Variables

You can define the port via environment variables, but the endpoint "/webhook" is static.

The following environment variables are available in Docker.  They will default to the values listed below.  YOU MUST DEFINE PLEXTOKEN AND PLEXSERVER IF USING PLEX WEBHOOKS!
| Variable              | Default Value | Description                                                                                                                                                                              |
|-----------------------|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| WHISPER_MODEL         | medium        | this can be tiny, base, small, medium, large                                                                                                                                             |
| WHISPER_THREADS       | 4             | number of threads to use during computation                                                                                                                                              |
| PROCADDEDMEDIA        | True          | will gen subtitles for all media added regardless of existing external/embedded subtitles (based off of SKIPIFINTERNALSUBLANG)                                                           |
| PROCMEDIAONPLAY       | True          | will gen subtitles for all played media regardless of existing external/embedded subtitles (based off of SKIPIFINTERNALSUBLANG)                                                          |
| NAMESUBLANG           | 'aa'            | allows you to pick what it will name the subtitle. Instead of using EN, I'm using AA, so it doesn't mix with exiting external EN subs, and AA will populate higher on the list in Plex. |
| SKIPIFINTERNALSUBLANG | 'eng'           | Will not generate a subtitle if the file has an internal sub matching the 3 letter code of this variable (See https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)                     |
| PLEXSERVER | 'http://plex:32400' | This needs to be set to your local plex server address/port |
| PLEXTOKEN | token here | This needs to be set to your plex token found by https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/ |
| WEBHOOKPORT | 8090 | Change this if you need a different port for your webhook |
| NEW OPTIONS AS OF 22 Oct 2023 | |
| CONCURRENT_TRANSCRIPTIONS | 2 | Number of files it will transcribe in parallel |
| TRANSCRIBE_DEVICE | 'cpu' | Can transcribe via gpu (Cuda only) or cpu.  Takes option of "cpu", "gpu", "cuda" |
| WORD_LEVEL_HIGHLIGHT | False | Highlights each words as it's spoken in the subtitle.  See example video @ https://github.com/jianfch/stable-ts |
| DEBUG | False | Provides some debug data that can be helpful to troubleshoot path mapping and other issues |
| USE_PATH_MAPPING | False | Similar to sonarr and radarr path mapping, this will attempt to replace paths on file systems that don't have identical paths.  Currently only support for one path replacement. Examples below. |
| PATH_MAPPING_FROM | '/tv' | This is the path of my media relative to my Plex server |
| PATH_MAPPING_TO | '/Volumes/TV' | This is the path of that same folder relative to my Mac Mini that will run the script |
| STORE_LOCAL_LIBS | 'True' | This will save and install the python libraries to the folder 'libs' in the same directory as the execution path of subgen.py.  This is primarily so you can manage the libs outside of a docker container if you desire. |

## Docker Volumes

You MUST mount your media volumes in subgen the same way Plex sees them.  For example, if Plex uses "/Share/media/TV:/tv" you must have that identical volume in subgen.  

"${APPDATA}/subgen:/whisper.cpp" is just for storage of the cloned and compiled code, also the models are stored in the /whisper.cpp/models, so it will prevent redownloading them.  This volume isn't necessary, just a nicety.  

## Running without Docker

You might have to tweak the script a little bit, but will work just fine without Docker.  You can either set the variables as environment variables in your CLI or edit the script manually at the top.  As mentioned above, your paths still have to match Plex. 

Example of instructions if you're on a Debian based linux once you set your environment variables:
```sh
wget https://raw.githubusercontent.com/McCloudS/subgen/main/subgen/subgen.py
apt-get update && apt-get install -y ffmpeg git gcc python3
pip3 install webhook_listener
python3 -u subgen_nodocker.py
```

# What are the limitations/problems?

* If Plex adds multiple shows (like a season pack), it will fail to process subtitles.  It is reliant on a SINGLE file to accurately work now.
* Long pauses/silence behave strangely.  It will likely show the previous or next words during long gaps of silence.  
* I made it and know nothing about formal deployment for python coding.  
* There is no 'wrapper' for python for whisper.cpp at this point, so I'm just using subprocess.call
* The Whisper.cpp/OpenAI model seems to fail in cases.  I've seen 1 or 2 instances where the subtitle will repeat the same line for several minutes.
* It's using trained AI models to transcribe, so it WILL mess up...but we find the transcription goofs amusing.  

# What's next?  

I'm hoping someone that is much more skilled than I, to use this as a pushing off point to make this better.  In a perfect world, this would integrate with Plex, Sonarr, Radarr, or Bazarr.  Bazarr tracks failed subtitle downloads, I originally wanted to utilize its API, but decided on my current solution for simplicity.  

Optimizations I can think of off hand:
* On played, use a faster model with speedup, since you might want those pretty quickly
* Fix processing for when adding multiple files
* There might be an OpenAI native CPU version now?  If so, it might be better since it's natively in python
* Cleaner implementation in a different language.  Python isn't the best for this particular implementation, but I wanted to learn it
* Whisper (.cpp) has the ability to translate a good chunk of languages into english.  I didn't explore this.  I'm not sure what this looks like with bi-lingual shows like Acapulco.  
* Add an ability via a web-ui or something to generate subtitles for particular media files/folders.

Will I update or maintain this?  Likely not.  I built this for my own use, and will fix and push issues that directly impact my own usage.  Unfortunately, I don't have the time or expertise to manage a project like this.  

# Additional reading:

* https://github.com/ggerganov/whisper.cpp/issues/89 (Benchmarks)
* https://github.com/openai/whisper/discussions/454 (Whisper CPU implementation)
* https://github.com/openai/whisper (Original OpenAI project)
* https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes (2 letter subtitle codes)

# Credits:  
* Whisper.cpp (https://github.com/ggerganov/whisper.cpp)
* Google
* ffmpeg
