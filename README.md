[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate/?hosted_button_id=SU4QQP6LH5PF6)

Howdy all,

This is a project I've had running for a bit, then cleaned up for 'release' while the kids were sleeping.  It's more of a POC, piece of crap, or a proof of concept.  This was also my first ever Python usage.

# BLUF:  Someone else use this idea (not the code!) as a jumping off point.

# What is this?

This is a half-assed attempt of transcribing subtitles (.srt) from your personal media in a Plex server using a CPU.  It is currently reliant on Tautulli for webhooks from Plex.  Why? During my limited testing, Plex was VERY sporadically actually sending out their webhooks using their built-in functionality (https://support.plex.tv/articles/115002267687-webhooks).  Tautulli gave a little bit more functionality, and for my use case, didn't require doing a bunch of Plex API calls because their webhooks had all the functionality I needed.  This uses whisper.cpp which is an implementation of OpenAI's Whisper model to use CPUs (Do your own research!).  While CPUs obviously aren't super efficient at this, but my server sits idle 99% of the time, so this worked great for me.  

# Why?

Honestly, I built this for me, but saw the utility in other people maybe using it.  This works well for my use case.  Since having children, I'm either deaf or wanting to have everything quiet.  We watch EVERYTHING with subtitles now, and I feel like I can't even understand the show without them.  I use Bazarr to auto-download, and gap fill with Plex's built-in capability.  This is for everything else.  Some shows just won't have subtitles available for some reason or another, or in some cases on my H265 media, they are wildly out of sync. 

# What can it do?

* Create .srt subtitles when a SINGLE media file is added or played via Plex which triggers off of Tautulli webhooks.  

# How do I set it up?

You need a working Tautulli installation linked to your Plex.   Can it be run without Docker?  Yes.  [See below](https://github.com/McCloudS/subgen/blob/main/README.md#running-without-docker)

You can now pull the image directly from Dockerhub:
```
docker pull mccloud/subgen
```
Or, you can build from scratch:
```
git clone https://github.com/mcclouds/subgen
cd subgen
docker-compose build
docker-compose up -d
```

Create the webhooks in Tautulli with the following settings:

Webhook URL: http://yourdockerip:8090

Webhook Method: Post

Triggers: Whatever you want, but you'll likely want "Playback Start" and "Recently Added"

Data: Under Playback Start, JSON Headers will be blank, JSON Data will be:
```json
{
            "event":"played",
            "file":"{file}",
            "filename":"{filename}",
            "mediatype":"{media_type}"
}
```

Similarly, under Recently Added: 
```json
{
            "event":"added",
            "file":"{file}",
            "filename":"{filename}",
            "mediatype":"{media_type}"
}
```
It should look like:

![Alt text](WebhookSettings.png?raw=true "Webhook Settings")

The following environment variables are optional in Docker-compose.  They will default to the values listed below.
| Variable              | Default Value | Description                                                                                                                                                                              |
|-----------------------|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| WHISPER_MODEL         | medium        | this can be tiny, base, small, medium, large                                                                                                                                             |
| WHISPER_SPEEDUP       | False         | this adds the option -su "speed up audio by x2 (reduced accuracy)"                                                                                                                       |
| WHISPER_THREADS       | 4             | number of threads to use during computation                                                                                                                                              |
| WHISPER_PROCESSORS    | 1             | number of processors to use during computation                                                                                                                                           |
| PROCADDEDMEDIA        | True          | will gen subtitles for all media added regardless of existing external/embedded subtitles (based off of SKIPIFINTERNALSUBLANG)                                                           |
| PROCMEDIAONPLAY       | False         | will gen subtitles for all played media regardless of existing external/embedded subtitles (based off of SKIPIFINTERNALSUBLANG)                                                          |
| NAMESUBLANG           | aa            | allows you to pick what it will name the subtitle. Instead of using EN, I'm using AA, so it doesn't mix with exiting external EN subs, and AA will populate higher on the list  in Plex. |
| UPDATEREPO            | True          | pulls and merges whisper.cpp on every start                                                                                                                                              |
| SKIPIFINTERNALSUBLANG | eng           | Will not generate a subtitle if the file has an internal sub matching the 3 letter code of this variable (See https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)                     |
## Docker Volumes

You MUST mount your media volumes in subgen the same way Plex sees them.  For example, if Plex uses "/Share/media/TV:/tv" you must have that identical volume in subgen.  

"${APPDATA}/subgen:/whisper.cpp" is just for storage of the cloned and compiled code, also the models are stored in the /whisper.cpp/models, so it will prevent redownloading them.  This volume isn't necessary, just a nicety.  

## Running without Docker

You might have to tweak the script a little bit, but will work just fine without Docker.  You can either set the variables as environment variables in your CLI or edit the script manually at the top.  As mentioned above, your paths still have to match Plex. 

Example of instructions if you're on a Debian based linux:
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
* Move it to a different API/Webhook
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
* Webhook_listener (https://pypi.org/project/Webhook-Listener)
* Tautulli (https://tautulli.com)
* Google
* ffmpeg
