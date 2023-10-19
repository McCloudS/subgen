import subprocess

# List of packages to install
packages_to_install = [
    'flask',
    'stable_ts',
    'requests',
    'faster-whisper',
]

for package in packages_to_install:
    # Run pip install command to install the package locally
    subprocess.run(['pip', 'install', package, '--target', 'libs'])

import os
import json
import xml.etree.ElementTree as ET
import threading
import av
import sys
    
def convert_to_bool(in_bool):
    value = str(in_bool).lower()
    return value not in ('false', 'off', '0')
    
sys.path.append('libs')

from flask import Flask, request    
import stable_whisper
import requests

# Replace your getenv calls with appropriate default values here
plextoken = os.getenv('PLEXTOKEN', "token here")
plexserver = os.getenv('PLEXSERVER', "http://192.168.1.111:32400")
whisper_model = os.getenv('WHISPER_MODEL', "medium")
whisper_threads = int(os.getenv('WHISPER_THREADS', 4))
concurrent_transcriptions = int(os.getenv('CONCURRENT_TRANSCRIPTIONS', '1'))
procaddedmedia = convert_to_bool(os.getenv('PROCADDEDMEDIA', "True"))
procmediaonplay = convert_to_bool(os.getenv('PROCMEDIAONPLAY', "True"))
namesublang = os.getenv('NAMESUBLANG', "aa")
skipifinternalsublang = os.getenv('SKIPIFINTERNALSUBLANG', "eng")
webhookport = int(os.getenv('WEBHOOKPORT', 8090))
word_level_highlight = convert_to_bool(os.getenv('WORD_LEVEL_HIGHLIGHT', "False"))
debug = convert_to_bool(os.getenv('DEBUG', False))
use_path_mapping = convert_to_bool(os.getenv('USE_PATH_MAPPING', False))
path_mapping_from = os.getenv('PATH_MAPPING_FROM', '/tv')
path_mapping_to = os.getenv('PATH_MAPPING_TO', '/Volumes/TV')

app = Flask(__name__)
model = stable_whisper.load_faster_whisper(whisper_model, cpu_threads=whisper_threads)
files_to_transcribe = set()
subextension =  '.subgen.' + whisper_model + '.' + namesublang + '.srt'

@app.route("/webhook", methods=["POST"])
def receive_webhook():
    if debug:
        print("We got a hook, let's figure out where it came from!")
    if request.headers.get("source") == "Tautulli":
        payload = request.json
        if debug:
            print("This hook is from Tautulli!")
    else:
        payload = json.loads(request.form['payload'])
    event = payload.get("event")
    if debug:
        print("event hook: " + str(payload))
    if ((event == "library.new" or event == "added") and procaddedmedia) or ((event == "media.play" or event == "played") and procmediaonplay):
        if event == "library.new" or event == "media.play": # these are the plex webhooks!
            print("This hook is from Plex!")
            fullpath = get_file_name(payload.get("Metadata").get("ratingKey"), plexserver, plextoken)
        elif event == "added" or event == "played":
            print("Tautulli webhook received!")
            fullpath = payload.get("file")
        else:
            print("Didn't get a webhook we expected, discarding")
            return ""

        print("Path of file: " + fullpath)
        if use_path_mapping:
            fullpath = fullpath.replace(path_mapping_from, path_mapping_to)
            print("Updated path: " + fullpath.replace(path_mapping_from, path_mapping_to))
        print("event: " + event)
        print("Transcriptions are limited to running " + str(concurrent_transcriptions) + " at a time")
        print("Running " + str(whisper_threads) + " threads per transcription")
                
        add_file_for_transcription(fullpath)

    return ""

def gen_subtitles(inputvideo):
    try:
        print(f"Transcribing file: {inputvideo}")
        result = model.transcribe_stable(inputvideo)
        result.to_srt_vtt(inputvideo.rsplit('.', 1)[0] + ' ' + subextension, word_level=word_level_highlight)
        print(f"Transcription of {file_path} is completed.")
        files_to_transcribe.remove(inputvideo)
    except Exception as e:
        print(f"Error processing or transcribing {file_path}: {e}")

# Function to add a file for transcription
def add_file_for_transcription(file_path):
    if file_path not in files_to_transcribe:
        
        if has_subtitle_language(file_path, skipifinternalsublang):
            print("File already has an internal sub we want, skipping generation")
            return "File already has an internal sub we want, skipping generation"
        elif os.path.exists(file_path.rsplit('.', 1)[0] + ' ' + subextension):
            print("We already have a subgen created for this file, skipping it")
            return "We already have a subgen created for this file, skipping it"
            
        files_to_transcribe.add(file_path)
        print(f"Added {file_path} for transcription.")
        # Start transcription for the file in a separate thread
    
        transcription_thread = threading.Thread(target=gen_subtitles, args=(file_path,))
        transcription_thread.start()
    else:
        print(f"File {file_path} is already in the transcription list. Skipping.")

def has_subtitle_language(video_file, target_language):
    try:
        container = av.open(video_file)
        subtitle_stream = None

        # Iterate through the streams in the video file
        for stream in container.streams:
            if stream.type == 'subtitle':
                # Check if the subtitle stream has the target language
                if 'language' in stream.metadata and stream.metadata['language'] == target_language:
                    subtitle_stream = stream
                    break

        if subtitle_stream:
            print(f"Subtitles in '{target_language}' language found in the video.")
            return True
        else:
            print(f"No subtitles in '{target_language}' language found in the video.")

        container.close()
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    
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

print("Starting webhook!")
if __name__ == "__main__":
    app.run(debug=debug, host='0.0.0.0', port=int(webhookport))
