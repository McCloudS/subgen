import sys
import os
import time
import json
import glob
import pathlib
import requests
import subprocess
from flask import Flask, request
import xml.etree.ElementTree as ET
        
def converttobool(in_bool):
    value = str(in_bool).lower()
    if value in ('false', 'off', '0'):
        return False
    else:
        return True

# parse our arguments from environment variables
plextoken = os.getenv('PLEXTOKEN', "tokenhere")
plexserver = os.getenv('PLEXSERVER', "http://plex:32400")
whisper_model = os.getenv('WHISPER_MODEL', "medium")
whisper_speedup = converttobool(os.getenv('WHISPER_SPEEDUP', "False"))
whisper_threads = os.getenv('WHISPER_THREADS', "4")
whisper_processors = os.getenv('WHISPER_PROCESSORS', "1")
procaddedmedia = converttobool(os.getenv('PROCADDEDMEDIA', "True"))
procmediaonplay = converttobool(os.getenv('PROCMEDIAONPLAY', "False"))
namesublang = os.getenv('NAMESUBLANG', "aa")
updaterepo = converttobool(os.getenv('UPDATEREPO', "True"))
skipifinternalsublang = os.getenv('SKIPIFINTERNALSUBLANG', "eng")
webhookport = os.getenv('WEBHOOKPORT', 8090)

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def receive_webhook():
    if request.headers.get("source") == "Tautulli":
        payload = request.json
    else:
        payload = json.loads(request.form['payload'])
    event = payload.get("event")
    if event != "library.new" and event != "added" and event != "media.play" and event != "played": return ""
    if ((event == "library.new" or event == "added") and procaddedmedia) or ((event == "media.play" or event == "played") and procmediaonplay):

        if event == "library.new" or event == "media.play": # these are the plex webhooks!
            print("Plex webhook received!")
            metadata = payload.get("Metadata")
            ratingkey = metadata.get("ratingKey")
            fullpath = get_file_name(ratingkey, plexserver, plextoken)
        elif event == "added" or event == "played":
            print("Tautulli webhook received!")
            fullpath = payload.get("file")
        else:
            print("Didn't get a webhook we expected, discarding")
            return ""
        
        filename = pathlib.Path(fullpath).name
        filepath = os.path.dirname(fullpath)
        filenamenoextension = filename.replace(pathlib.Path(fullpath).suffix, "")

        print("fullpath: " + fullpath)
        print("filepath: " + filepath)
        print("file name with no extension: " + filenamenoextension)
        print("event: " + event)
    
        if skipifinternalsublang in str(subprocess.check_output("ffprobe -loglevel error -select_streams s -show_entries stream=index:stream_tags=language -of csv=p=0 \"{}\"".format(fullpath), shell=True)):
            print("File already has an internal sub we want, skipping generation")
            return "File already has an internal sub we want, skipping generation"
        elif os.path.isfile("{}.output.wav".format(fullpath)):
            print("WAV file already exists, we're assuming it's processing and skipping it")
            return "WAV file already exists, we're assuming it's processing and skipping it"
        elif len(glob.glob("{}/{}*subgen*".format(filepath, filenamenoextension))) > 0:
            print("We already have a subgen created for this file, skipping it")
            return "We already have a subgen created for this file, skipping it"
           
        if whisper_speedup:
            print("This is a speedup run!")
            print(whisper_speedup)
            finalsubname = "{0}/{1}.subgen.{2}.speedup.{3}".format(filepath, filenamenoextension, whisper_model, namesublang)
        else:
            print("No speedup")
            finalsubname = "{0}/{1}.subgen.{2}.{3}".format(filepath, filenamenoextension, whisper_model, namesublang)
                
        gen_subtitles(fullpath, "{}.output.wav".format(fullpath), finalsubname)
  
        if os.path.isfile("{}.output.wav".format(fullpath)):
            print("Deleting WAV workfile")
            os.remove("{}.output.wav".format(fullpath))

    return ""

def gen_subtitles(filename, inputwav, finalsubname):
    strip_audio(filename)
    run_whisper(inputwav, finalsubname)

def strip_audio(filename):
    print("Starting strip audio")
    command = "ffmpeg -y -i \"{}\" -ar 16000 -ac 1 -c:a pcm_s16le \"{}.output.wav\"".format(
        filename, filename)
    print("Command: " + command)
    subprocess.call(command, shell=True)
    print("Done stripping audio")

def run_whisper(inputwav, finalsubname):
    print("Starting whisper")
    os.chdir("/whisper.cpp")
    command = "./main -m models/ggml-{}.bin -of \"{}\" -t {} -p {} -osrt -f \"{}\"" .format(
        whisper_model, finalsubname, whisper_threads, whisper_processors, inputwav)
    if (whisper_speedup):
        command = command.replace("-osrt", "-osrt -su")
    print("Command: " + command)
    subprocess.call(command, shell=True)

    print("Done with whisper")
    
def get_file_name(item_id, plexserver, plextoken):
    url = f"{plexserver}/library/metadata/{item_id}"

    response = requests.get(url, headers={
      "X-Plex-Token": "{plextoken}"})

    if response.status_code == 200:
        root = ET.fromstring(response.text)
        fullpath = root.find(".//Part").attrib['file']
        return fullpath
    else:
        print(f"Error: {response.text}")
    return


if not os.path.isdir("/whisper.cpp"):
    os.mkdir("/whisper.cpp")
os.chdir("/whisper.cpp")
subprocess.call("git clone https://github.com/ggerganov/whisper.cpp .", shell=True)
if updaterepo:
    print("Updating repo!")
    #subprocess.call("git pull", shell=True)
if os.path.isfile("/whisper.cpp/samples/jfk.wav"): # delete the sample file, so it doesn't try transcribing it.  Saves us a couple seconds.
    print("Deleting sample file")
    #os.remove("/whisper.cpp/samples/jfk.wav")
subprocess.call("make " + whisper_model, shell=True)
print("Starting webhook!")
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(webhookport))
