import subprocess
import os
import json
import xml.etree.ElementTree as ET
import threading
import sys
import time
import queue
import logging
import gc
from array import array
from typing import Union, Any

# List of packages to install
packages_to_install = [
    'numpy',
    'stable-ts',
    'fastapi',
    'requests',
    'faster-whisper',
    'uvicorn',
    'python-multipart',
    # Add more packages as needed
]

for package in packages_to_install:
    print(f"Installing {package}...")
    try:
        subprocess.run(['pip3', 'install', package], check=True)
        print(f"{package} has been successfully installed.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")

from fastapi import FastAPI, File, UploadFile, Query, Header, Body, Form, Request
from fastapi.responses import StreamingResponse, RedirectResponse    
import stable_whisper
import requests
import av

def convert_to_bool(in_bool):
    if isinstance(in_bool, bool):
        return in_bool
    else:
        value = str(in_bool).lower()
        return value not in ('false', 'off', '0')

# Replace your getenv calls with appropriate default values here
plextoken = os.getenv('PLEXTOKEN', "token here")
plexserver = os.getenv('PLEXSERVER', "http://192.168.1.111:32400")
jellyfintoken = os.getenv('JELLYFINTOKEN', "token here")
jellyfinserver = os.getenv('JELLYFINSERVER', "http://192.168.1.111:8096")
whisper_model = os.getenv('WHISPER_MODEL', "medium")
whisper_threads = int(os.getenv('WHISPER_THREADS', 4))
concurrent_transcriptions = int(os.getenv('CONCURRENT_TRANSCRIPTIONS', '2'))
transcribe_device = os.getenv('TRANSCRIBE_DEVICE', "cpu")
procaddedmedia = convert_to_bool(os.getenv('PROCADDEDMEDIA', True))
procmediaonplay = convert_to_bool(os.getenv('PROCMEDIAONPLAY', True))
namesublang = os.getenv('NAMESUBLANG', "aa")
skipifinternalsublang = os.getenv('SKIPIFINTERNALSUBLANG', "eng")
webhookport = int(os.getenv('WEBHOOKPORT', 8090))
word_level_highlight = convert_to_bool(os.getenv('WORD_LEVEL_HIGHLIGHT', False))
debug = convert_to_bool(os.getenv('DEBUG', False))
use_path_mapping = convert_to_bool(os.getenv('USE_PATH_MAPPING', False))
path_mapping_from = os.getenv('PATH_MAPPING_FROM', '/tv')
path_mapping_to = os.getenv('PATH_MAPPING_TO', '/Volumes/TV')
model_location = os.getenv('MODEL_PATH', '.')
transcribe_folders = os.getenv('TRANSCRIBE_FOLDERS', '')
transcribe_or_translate = os.getenv('TRANSCRIBE_OR_TRANSLATE', 'translate')
if transcribe_device == "gpu":
    transcribe_device = "cuda"

app = FastAPI()
model = None
files_to_transcribe = []
subextension =  f".subgen.{whisper_model.split('.')[0]}.{namesublang}.srt"
print(f"Transcriptions are limited to running {str(concurrent_transcriptions)} at a time")
print(f"Running {str(whisper_threads)} threads per transcription")

if debug:
    logging.basicConfig(stream=sys.stderr, level=logging.NOTSET)
else:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

@app.post("/webhook")
def print_warning():
    print("*** This is the legacy webhook.  You need to update to webhook urls to end in plex, tautulli, or jellyfin instead of webhook. ***")
    return ""

@app.post("/tautulli")
def receive_tautulli_webhook(
    source: Union[str, None] = Header(None), 
    event: str = Body(None),
    file: str = Body(None),
    ):
    
    if source == "Tautulli":
        logging.debug(f"Tautulli event detected is: {event}")
        if((event == "added" and procaddedmedia) or (event == "played" and procmediaonplay)):
            fullpath = file
            logging.debug("Path of file: " + fullpath)
        
            gen_subtitles(path_mapping(fullpath), transcribe_or_translate, True)
    else:
        print("This doesn't appear to be a properly configured Tautulli webhook, please review the instructions again!")
    
    return ""
    
@app.post("/plex")
def receive_plex_webhook(
    user_agent: Union[str, None] = Header(None), 
    payload: Union[str, None] = Form(),
    ):
    plex_json = json.loads(payload)
    logging.debug(f"Raw response: {payload}")
    
    if "PlexMediaServer" in user_agent:
        event = plex_json["event"]
        logging.debug(f"Plex event detected is: {event}")
        if((event == "library.new" and procaddedmedia) or (event == "media.play" and procmediaonplay)):
            fullpath = get_plex_file_name(plex_json['Metadata']['ratingKey'], plexserver, plextoken)
            logging.debug("Path of file: " + fullpath)
     
            gen_subtitles(path_mapping(fullpath), transcribe_or_translate, True)
    else:
        print("This doesn't appear to be a properly configured Plex webhook, please review the instructions again!")
     
    return ""

@app.post("/jellyfin")
def receive_jellyfin_webhook(
    user_agent: Union[str, None] = Header(None), 
    NotificationType: str = Body(None),
    file: str = Body(None),
    ItemId: str = Body(None),
    ):
    
    if "Jellyfin-Server" in user_agent:
        logging.debug("Jellyfin event detected is: " + NotificationType)
        logging.debug("itemid is: " + ItemId)
        if((NotificationType == "ItemAdded" and procaddedmedia) or (NotificationType == "PlaybackStart" and procmediaonplay)):
            fullpath = get_jellyfin_file_name(ItemId, jellyfinserver, jellyfintoken)
            logging.debug(f"Path of file: {fullpath}")
     
            gen_subtitles(path_mapping(fullpath), transcribe_or_translate, True)
    else:
        print("This doesn't appear to be a properly configured Jellyfin webhook, please review the instructions again!")
     
    return ""

@app.post("/emby")
def receive_emby_webhook(
    user_agent: Union[str, None] = Header(None), 
    data: Union[str, None] = Form(None),
    ):
    logging.debug("Raw response: %s", data)
    
    if "Emby Server" in user_agent:
        if data:
            data_dict = json.loads(data)
            fullpath = data_dict['Item']['Path']
            event = data_dict['Event']
            logging.debug("Emby event detected is: " + event)
            if((event == "library.new" and procaddedmedia) or (event == "playback.start" and procmediaonplay)):
                logging.debug("Path of file: " + fullpath)
     
                gen_subtitles(path_mapping(fullpath), transcribe_or_translate, True)
    else:
        print("This doesn't appear to be a properly configured Emby webhook, please review the instructions again!")
     
    return ""

def gen_subtitles(file_path: str, transcribe_or_translate_str: str, front=True) -> None:
    """Generates subtitles for a video file.

    Args:
        file_path: The path to the video file.
        transcription_or_translation: The type of transcription or translation to perform.
        front: Whether to add the file to the front of the transcription queue.
    """
    global model
    
    try:
        if not is_video_file(file_path):
            print(f"{file_path} isn't a video file!")
            return None
            
        if file_path not in files_to_transcribe:
            if has_subtitle_language(file_path, skipifinternalsublang):
                logging.debug(f"{file_path} already has an internal sub we want, skipping generation")
                return f"{file_path} already has an internal sub we want, skipping generation"
            elif os.path.exists(file_path.rsplit('.', 1)[0] + subextension):
                print(f"{file_path} already has a subgen created for this, skipping it")
                return f"{file_path} already has a subgen created for this, skipping it"
                
            if front:
                files_to_transcribe.insert(0, file_path)
            else:
                files_to_transcribe.append(file_path)
            print(f"Added {os.path.basename(file_path)} for transcription.")
            # Start transcription for the file in a separate thread

            print(f"{len(files_to_transcribe)} files in the queue for transcription")
            print(f"Transcribing file: {os.path.basename(file_path)}")
            start_time = time.time()
            if model is None:
                logging.debug("Model was purged, need to re-create")
                model = stable_whisper.load_faster_whisper(whisper_model, download_root=model_location, device=transcribe_device, cpu_threads=whisper_threads, num_workers=concurrent_transcriptions)
            
            result = model.transcribe_stable(file_path, task=transcribe_or_translate_str)
            result.to_srt_vtt(file_path.rsplit('.', 1)[0] + subextension, word_level=word_level_highlight)
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            print(f"Transcription of {os.path.basename(file_path)} is completed, it took {minutes} minutes and {seconds} seconds to complete.")
            files_to_transcribe.remove(file_path)
        else:
            print(f"File {os.path.basename(file_path)} is already in the transcription list. Skipping.")

    except Exception as e:
        print(f"Error processing or transcribing {file_path}: {e}")
    finally:
        if len(files_to_transcribe) == 0:
            logging.debug("Queue is empty, clearing/releasing VRAM")
            try:
                del model
            except Exception as e:
                None
            gc.collect()

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
            logging.debug(f"Subtitles in '{target_language}' language found in the video.")
            return True
        else:
            logging.debug(f"No subtitles in '{target_language}' language found in the video.")

        container.close()
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    
def get_plex_file_name(itemid: str, server_ip: str, plex_token: str) -> str:
    """Gets the full path to a file from the Plex server.

    Args:
        itemid: The ID of the item in the Plex library.
        server_ip: The IP address of the Plex server.
        plex_token: The Plex token.

    Returns:
        The full path to the file.
    """

    url = f"{server_ip}/library/metadata/{itemid}"

    headers = {
        "X-Plex-Token": plex_token,
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        root = ET.fromstring(response.content)
        fullpath = root.find(".//Part").attrib['file']
        return fullpath
    else:
        raise Exception(f"Error: {response.status_code}")

def get_jellyfin_file_name(item_id: str, jellyfin_url: str, jellyfin_token: str) -> str:
    """Gets the full path to a file from the Jellyfin server.

    Args:
        jellyfin_url: The URL of the Jellyfin server.
        jellyfin_token: The Jellyfin token.
        item_id: The ID of the item in the Jellyfin library.

    Returns:
        The full path to the file.
    """

    headers = {
        "Authorization": f"MediaBrowser Token={jellyfin_token}",
    }

    # Cheap way to get the admin user id, and save it for later use.
    users = json.loads(requests.get(f"{jellyfin_url}/Users", headers=headers).content)
    jellyfin_admin = get_jellyfin_admin(users)

    response = requests.get(f"{jellyfin_url}/Users/{jellyfin_admin}/Items/{item_id}", headers=headers)

    if response.status_code == 200:
        file_name = json.loads(response.content)['Path']
        return file_name
    else:
        raise Exception(f"Error: {response.status_code}")

def get_jellyfin_admin(users):
    for user in users:
        if user["Policy"]["IsAdministrator"]:
            return user["Id"]
            
    raise Exception("Unable to find administrator user in Jellyfin")

def is_video_file(file_path):
    av.logging.set_level(av.logging.PANIC)
    try:
        container = av.open(file_path)
        for stream in container.streams:
            if stream.type == 'video':
                return True
        return False
    except av.AVError:
        return False

def path_mapping(fullpath):
    if use_path_mapping:
        fullpath = fullpath.replace(path_mapping_from, path_mapping_to)
        logging.debug("Updated path: " + fullpath.replace(path_mapping_from, path_mapping_to))
    return fullpath

def transcribe_existing():
    print("Starting to search folders to see if we need to create subtitles.")
    logging.debug("The folders are:")
    for path in transcribe_folders:
        logging.debug(path)
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                if is_video_file(file_path):
                    gen_subtitles(path_mapping(fullpath), transcribe_or_translate, False)
                    
    print("Finished searching and queueing files for transcription")
                    
if transcribe_folders:
    transcribe_folders = transcribe_folders.split(",")
    transcribe_existing()

print("Starting webhook!")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("subgen:app", host="0.0.0.0", port=int(webhookport), reload=debug, use_colors=True)
