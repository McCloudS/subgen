[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate/?hosted_button_id=SU4QQP6LH5PF6)

Updates:

23 Oct 2023: There are now two docker images, ones for CPU (it's smaller): mccloud/subgen:latest, mccloud/subgen:cpu, the other is for cuda/GPU: mccloud/subgen:cuda

22 Oct 2023: The script should have backwards compability with previous envirionment settings, but just to be sure, look at the new options below.  If you don't want to manually edit your environment variables, just edit the script manually. While I have added GPU support, I haven't tested it yet.

19 Oct 2023: And we're back!  Uses faster-whisper and stable-ts.  Shouldn't break anything from previous settings, but adds a couple new options that aren't documented at this point in time.  As of now, this is not a docker image on dockerhub.  The potential intent is to move this eventually to a pure python script, primarily to simplify my efforts.  Quick and dirty to meet dependencies: pip or `pip3 install flask requests stable-ts faster-whisper`

This potentially has the ability to use CUDA/Nvidia GPU's, but I don't have one set up yet.  Tesla T4 is in the mail!

2 Feb 2023: Added Tautulli webhooks back in.  Didn't realize Plex webhooks was PlexPass only.  See below for instructions to add it back in.

31 Jan 2023 : Rewrote the script substantially to remove Tautulli and fix some variable handling.  For some reason my implementation requires the container to be in host mode.  My Plex was giving "401 Unauthorized" when attempt to query from docker subnets during API calls.

Howdy all,

This is a project I've had running for a bit, then cleaned up for 'release' while the kids were sleeping.  It's more of a POC, piece of crap, or a proof of concept.  This was also my first ever Python usage.

# BLUF:  Someone else use this idea (not the code!) as a jumping off point.

# What is this?

This is a half-assed attempt of transcribing subtitles (.srt) from your personal media in a Plex server.  It is currently reliant on webhooks from Plex or Tautulli.  This uses stable-ts and faster-whisper which can use both Nvidia GPUs and CPUs.  While CPUs obviously aren't super efficient at this, my server sits idle 99% of the time, so this worked great for me.  

# Why?

Honestly, I built this for me, but saw the utility in other people maybe using it.  This works well for my use case.  Since having children, I'm either deaf or wanting to have everything quiet.  We watch EVERYTHING with subtitles now, and I feel like I can't even understand the show without them.  I use Bazarr to auto-download, and gap fill with Plex's built-in capability.  This is for everything else.  Some shows just won't have subtitles available for some reason or another, or in some cases on my H265 media, they are wildly out of sync. 

# What can it do?

* Create .srt subtitles when a SINGLE media file is added or played via Plex which triggers off of Plex or Tautulli webhooks. 

# How do I set it up?

Install python3 and execute the script.  You need to have matching paths relative to your Plex server/folders, or use USE_PATH_MAPPING.  The dockerfile is also posted on dockerhub (mccloud/subgen)

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
| WHISPER_MODEL         | 'medium'        | this can be tiny, base, small, medium, large                                                                                                                                             |
| WHISPER_THREADS       | 4             | number of threads to use during computation                                                                                                                                              |
| PROCADDEDMEDIA        | True          | will gen subtitles for all media added regardless of existing external/embedded subtitles (based off of SKIPIFINTERNALSUBLANG)                                                           |
| PROCMEDIAONPLAY       | True          | will gen subtitles for all played media regardless of existing external/embedded subtitles (based off of SKIPIFINTERNALSUBLANG)                                                          |
| NAMESUBLANG           | 'aa'            | allows you to pick what it will name the subtitle. Instead of using EN, I'm using AA, so it doesn't mix with exiting external EN subs, and AA will populate higher on the list in Plex. |
| SKIPIFINTERNALSUBLANG | 'eng'           | Will not generate a subtitle if the file has an internal sub matching the 3 letter code of this variable (See https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)                     |
| PLEXSERVER | 'http://plex:32400' | This needs to be set to your local plex server address/port |
| PLEXTOKEN | 'token here' | This needs to be set to your plex token found by https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/ |
| WEBHOOKPORT | 8090 | Change this if you need a different port for your webhook |
| NEW OPTIONS AS OF 22 Oct 2023 | |
| CONCURRENT_TRANSCRIPTIONS | 2 | Number of files it will transcribe in parallel |
| TRANSCRIBE_DEVICE | 'cpu' | Can transcribe via gpu (Cuda only) or cpu.  Takes option of "cpu", "gpu", "cuda" |
| WORD_LEVEL_HIGHLIGHT | False | Highlights each words as it's spoken in the subtitle.  See example video @ https://github.com/jianfch/stable-ts |
| DEBUG | False | Provides some debug data that can be helpful to troubleshoot path mapping and other issues. Fun fact, if this is set to true, any modifications to the script will auto-reload it (if it isn't actively transcoding).  Useful to make small tweaks without re-downloading the whole file. |
| USE_PATH_MAPPING | False | Similar to sonarr and radarr path mapping, this will attempt to replace paths on file systems that don't have identical paths.  Currently only support for one path replacement. Examples below. |
| PATH_MAPPING_FROM | '/tv' | This is the path of my media relative to my Plex server |
| PATH_MAPPING_TO | '/Volumes/TV' | This is the path of that same folder relative to my Mac Mini that will run the script |

## Docker Configuration

You MUST mount your media volumes in subgen the same way Plex sees them.  For example, if Plex uses "/Share/media/TV:/tv" you must have that identical volume in subgen.  

`"${APPDATA}/subgen:/subgen"` is just for storage of the python script and the language model, so it will prevent redownloading them.  This volume isn't necessary, just a nicety. <br><br>`"${APPDATA}/subgen/dist-packages:/usr/local/lib/python3.10/dist-packages"` is just for storing of the python packages, so it won't redownload then, again, not necessary.

If you want to use a GPU, you need to map it accordingly.  

### Images:
mccloud/subgen:latest or mccloud/subgen: cpu is CPU only (smaller)<br>
mccloud/subgen:cuda is for GPU support
<br><br>
They are functionally identical, except the cuda variant has all the necessary dependancies for using a GPU.

## Running without Docker

The script was re-developed without using Docker, so it will work just fine as long as you have the dependencies installed.  You can either set the variables as environment variables in your CLI or edit the script manually at the top.  As mentioned above, your paths still have to match Plex. 

# What are the limitations/problems?

* If Plex adds multiple shows (like a season pack), it may fail to process subtitles.  It is reliant on a SINGLE file to accurately work now.
* Long pauses/silence behave strangely.  It will likely show the previous or next words during long gaps of silence.  (**This is mostly fixed now using stable-ts!**)
* I made it and know nothing about formal deployment for python coding.  
* It's using trained AI models to transcribe, so it WILL mess up...but we find the transcription goofs amusing.  

# What's next?  

I'm hoping someone that is much more skilled than I, to use this as a pushing off point to make this better.  In a perfect world, this would integrate with Plex, Sonarr, Radarr, or Bazarr.  Bazarr tracks failed subtitle downloads, I originally wanted to utilize its API, but decided on my current solution for simplicity.  

Optimizations I can think of off hand:
* Fix processing for when adding multiple files
* Whisper has the ability to translate a good chunk of languages into english.  I didn't explore this.  I'm not sure what this looks like with bi-lingual shows like Acapulco.  
* Add an ability via a web-ui or something to generate subtitles for particular media files/folders.

Will I update or maintain this?  Likely not.  I built this for my own use, and will fix and push issues that directly impact my own usage.  Unfortunately, I don't have the time or expertise to manage a project like this.  

# Additional reading:

* https://github.com/ggerganov/whisper.cpp/issues/89 (Benchmarks)
* https://github.com/openai/whisper/discussions/454 (Whisper CPU implementation)
* https://github.com/openai/whisper (Original OpenAI project)
* https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes (2 letter subtitle codes)

# Credits:  
* Whisper.cpp (https://github.com/ggerganov/whisper.cpp) for original implementation
* Google
* ffmpeg
* https://github.com/jianfch/stable-ts
* https://github.com/guillaumekln/faster-whisper
