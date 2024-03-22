subgen_version = '2024.3.22.52'

from datetime import datetime
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
import io
import random
from typing import BinaryIO, Union, Any
from fastapi import FastAPI, File, UploadFile, Query, Header, Body, Form, Request
from fastapi.responses import StreamingResponse, RedirectResponse
import numpy as np
import stable_whisper
from stable_whisper import Segment
import requests
import av
import ffmpeg
import whisper
import re

def convert_to_bool(in_bool):
    # Convert the input to string and lower case, then check against true values
    return str(in_bool).lower() in ('true', 'on', '1', 'y', 'yes')

# Replace your getenv calls with appropriate default values here
plextoken = os.getenv('PLEXTOKEN', 'token here')
plexserver = os.getenv('PLEXSERVER', 'http://192.168.1.111:32400')
jellyfintoken = os.getenv('JELLYFINTOKEN', 'token here')
jellyfinserver = os.getenv('JELLYFINSERVER', 'http://192.168.1.111:8096')
whisper_model = os.getenv('WHISPER_MODEL', 'medium')
whisper_threads = int(os.getenv('WHISPER_THREADS', 4))
concurrent_transcriptions = int(os.getenv('CONCURRENT_TRANSCRIPTIONS', 2))
transcribe_device = os.getenv('TRANSCRIBE_DEVICE', 'cpu')
procaddedmedia = convert_to_bool(os.getenv('PROCADDEDMEDIA', True))
procmediaonplay = convert_to_bool(os.getenv('PROCMEDIAONPLAY', True))
namesublang = os.getenv('NAMESUBLANG', 'aa')
skipifinternalsublang = os.getenv('SKIPIFINTERNALSUBLANG', 'eng')
webhookport = int(os.getenv('WEBHOOKPORT', 9000))
word_level_highlight = convert_to_bool(os.getenv('WORD_LEVEL_HIGHLIGHT', False))
debug = convert_to_bool(os.getenv('DEBUG', True))
use_path_mapping = convert_to_bool(os.getenv('USE_PATH_MAPPING', False))
path_mapping_from = os.getenv('PATH_MAPPING_FROM', r'/tv')
path_mapping_to = os.getenv('PATH_MAPPING_TO', r'/Volumes/TV')
model_location = os.getenv('MODEL_PATH', './models')
monitor = convert_to_bool(os.getenv('MONITOR', False))
transcribe_folders = os.getenv('TRANSCRIBE_FOLDERS', '')
transcribe_or_translate = os.getenv('TRANSCRIBE_OR_TRANSLATE', 'transcribe')
force_detected_language_to = os.getenv('FORCE_DETECTED_LANGUAGE_TO', '')
clear_vram_on_complete = convert_to_bool(os.getenv('CLEAR_VRAM_ON_COMPLETE', True))
compute_type = os.getenv('COMPUTE_TYPE', 'auto')
append = convert_to_bool(os.getenv('APPEND', False))
reload_script_on_change = convert_to_bool(os.getenv('RELOAD_SCRIPT_ON_CHANGE', False))
model_prompt = os.getenv('USE_MODEL_PROMPT', 'False')
custom_model_prompt = os.getenv('CUSTOM_MODEL_PROMPT', '')
lrc_for_audio_files = convert_to_bool(os.getenv('LRC_FOR_AUDIO_FILES', True))

if transcribe_device == "gpu":
    transcribe_device = "cuda"

if monitor:
    from watchdog.observers.polling import PollingObserver as Observer
    from watchdog.events import FileSystemEventHandler

app = FastAPI()
model = None
files_to_transcribe = []
subextension =  f".subgen.{whisper_model.split('.')[0]}.{namesublang}.srt"
subextensionSDH =  f".subgen.{whisper_model.split('.')[0]}.{namesublang}.sdh.srt"

in_docker = os.path.exists('/.dockerenv')
docker_status = "Docker" if in_docker else "Standalone"
last_print_time = None

# Define a filter class
class MultiplePatternsFilter(logging.Filter):
    def filter(self, record):
        # Define the patterns to search for
        patterns = [
            "Compression ratio threshold is not met",
            "Processing segment at",
            "Log probability threshold is",
            "Reset prompt",
            "Attempting to release",
            "released on ",
            "Attempting to acquire",
            "acquired on",
            "header parsing failed",
            "timescale not set",
            "misdetection possible",
            "srt was added",
            "doesn't have any audio to transcribe",
        ]
        # Return False if any of the patterns are found, True otherwise
        return not any(pattern in record.getMessage() for pattern in patterns)

# Configure logging
if debug:
    level = logging.DEBUG
    logging.basicConfig(stream=sys.stderr, level=level, format="%(asctime)s %(levelname)s: %(message)s")
else:
    level = logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level)

# Get the root logger
logger = logging.getLogger()
logger.setLevel(level)  # Set the logger level

for handler in logger.handlers:
    handler.addFilter(MultiplePatternsFilter())

logging.getLogger("multipart").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)

#This forces a flush to print progress correctly
def progress(seek, total):
    sys.stdout.flush()
    sys.stderr.flush()
    if(docker_status) == 'Docker':
        global last_print_time
        # Get the current time
        current_time = time.time()
    
        # Check if 5 seconds have passed since the last print
        if last_print_time is None or (current_time - last_print_time) >= 5:
            # Update the last print time
            last_print_time = current_time
            # Log the message
            logging.info("Force Update...")

TIME_OFFSET = 5

def appendLine(result):
    if append:
        lastSegment = result.segments[-1]
        date_time_str = datetime.now().strftime("%d %b %Y - %H:%M:%S")
        appended_text = f"Transcribed by whisperAI with faster-whisper ({whisper_model}) on {date_time_str}"
        
        # Create a new segment with the updated information
        newSegment = Segment(
            start=lastSegment.start + TIME_OFFSET,
            end=lastSegment.end + TIME_OFFSET,
            text=appended_text,
            words=[],  # Empty list for words
            id=lastSegment.id + 1
        )
        
        # Append the new segment to the result's segments
        result.segments.append(newSegment)

@app.get("/plex")
@app.get("/webhook")
@app.get("/jellyfin")
@app.get("/asr")
@app.get("/emby")
@app.get("/detect-language")
@app.get("/tautulli")
@app.get("/")
def handle_get_request(request: Request):
    return {"You accessed this request incorrectly via a GET request.  See https://github.com/McCloudS/subgen for proper configuration"}

@app.get("/status")
def status():
    in_docker = os.path.exists('/.dockerenv')
    docker_status = "Docker" if in_docker else "Standalone"
    return {"version" : f"Subgen {subgen_version}, stable-ts {stable_whisper.__version__}, whisper {whisper.__version__} ({docker_status})"}

@app.post("/subsync")
def subsync(
        audio_file: UploadFile = File(...),
        subtitle_file: UploadFile = File(...),
        language: Union[str, None] = Query(default=None),
):
    try:
        logging.info(f"Syncing subtitle file from Subsync webhook")
        result = None
        
        srt_content = subtitle_file.file.read().decode('utf-8')
        srt_content = re.sub(r'\{.*?\}', '', srt_content)
        # Remove numeric counters for each entry
        srt_content = re.sub(r'^\d+$', '', srt_content, flags=re.MULTILINE)
        # Remove timestamps and formatting
        srt_content = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', srt_content)
        # Remove any remaining newlines and spaces
        srt_content = re.sub(r'\n\n+', '\n', srt_content).strip()
                
        start_time = time.time()
        start_model()

        result = model.align(audio_file.file.read(), srt_content, language=language)
        appendLine(result)
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        logging.info(f"Subsync is completed, it took {minutes} minutes and {seconds} seconds to complete.")
    except Exception as e:
        logging.info(f"Error processing or aligning {audio_file.filename} or {subtitle_file.filename}: {e}")
    finally:
        delete_model()
    if result:
        return StreamingResponse(
            iter(result.to_srt_vtt(filepath = None, word_level=word_level_highlight)),
            media_type="text/plain",
            headers={
                'Source': 'Aligned using stable-ts from Subgen!',
            })
    else:
        return

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
        return {"This doesn't appear to be a properly configured Tautulli webhook, please review the instructions again!"}
    
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
            try:
                refresh_plex_metadata(plex_json['Metadata']['ratingKey'], plexserver, plextoken)
                logging.info(f"Metadata for item {plex_json['Metadata']['ratingKey']} refreshed successfully.")
            except Exception as e:
                logging.error(f"Failed to refresh metadata for item {plex_json['Metadata']['ratingKey']}: {e}")
    else:
        return {"This doesn't appear to be a properly configured Plex webhook, please review the instructions again!"}
     
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
            try:
                refresh_jellyfin_metadata(ItemId, jellyfinserver, jellyfintoken)
                logging.info(f"Metadata for item {ItemId} refreshed successfully.")
            except Exception as e:
                logging.error(f"Failed to refresh metadata for item {ItemId}: {e}")
    else:
        return {"This doesn't appear to be a properly configured Jellyfin webhook, please review the instructions again!"}
     
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
        return {"This doesn't appear to be a properly configured Emby webhook, please review the instructions again!"}
     
    return ""
    
@app.post("/batch")
def batch(
        directory: Union[str, None] = Query(default=None),
        forceLanguage: Union[str, None] = Query(default=None)
):
    transcribe_existing(directory, forceLanguage)
    
# idea and some code for asr and detect language from https://github.com/ahmetoner/whisper-asr-webservice
@app.post("//asr")
@app.post("/asr")
def asr(
        task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
        language: Union[str, None] = Query(default=None),
        initial_prompt: Union[str, None] = Query(default=None),  #not used by Bazarr
        audio_file: UploadFile = File(...),
        encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),  #not used by Bazarr/always False
        output: Union[str, None] = Query(default="srt", enum=["txt", "vtt", "srt", "tsv", "json"]),
        word_timestamps: bool = Query(default=False, description="Word level timestamps") #not used by Bazarr
):
    try:
        logging.info(f"Transcribing file from Bazarr/ASR webhook")
        result = None
        random_name = random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890", k=6)
        
        start_time = time.time()
        start_model()
        files_to_transcribe.insert(0, f"Bazarr-asr-{random_name}")
        audio_data = np.frombuffer(audio_file.file.read(), np.int16).flatten().astype(np.float32) / 32768.0
        if(model_prompt):
            custom_model_prompt = greetings_translations.get(language, '') or custom_model_prompt
        result = model.transcribe_stable(audio_data, task=task, input_sr=16000, language=language, progress_callback=progress, initial_prompt=custom_model_prompt)
        appendLine(result)
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        logging.info(f"Bazarr transcription is completed, it took {minutes} minutes and {seconds} seconds to complete.")
    except Exception as e:
        logging.info(f"Error processing or transcribing Bazarr {audio_file.filename}: {e}")
    finally:
        if f"Bazarr-asr-{random_name}" in files_to_transcribe:
            files_to_transcribe.remove(f"Bazarr-asr-{random_name}")
        delete_model()
    if result:
        return StreamingResponse(
            iter(result.to_srt_vtt(filepath = None, word_level=word_level_highlight)),
            media_type="text/plain",
            headers={
                'Source': 'Transcribed using stable-ts from Subgen!',
            })
    else:
        return

@app.post("//detect-language")
@app.post("/detect-language")
def detect_language(
        audio_file: UploadFile = File(...),
        #encode: bool = Query(default=True, description="Encode audio first through ffmpeg") # This is always false from Bazarr
):    
    detected_lang_code = ""  # Initialize with an empty string
    try:
        start_model()
        random_name = random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890", k=6)
        files_to_transcribe.insert(0, f"Bazarr-detect-language-{random_name}")
        audio_data = np.frombuffer(audio_file.file.read(), np.int16).flatten().astype(np.float32) / 32768.0
        detected_lang_code = model.transcribe_stable(whisper.pad_or_trim(audio_data), input_sr=16000).language
            
    except Exception as e:
        logging.info(f"Error processing or transcribing Bazarr {audio_file.filename}: {e}")
        
    finally:
        if f"Bazarr-detect-language-{random_name}" in files_to_transcribe:
            files_to_transcribe.remove(f"Bazarr-detect-language-{random_name}")
        delete_model()

        return {"detected_language": whisper_languages.get(detected_lang_code, detected_lang_code) , "language_code": detected_lang_code}

def start_model():
    global model
    if model is None:
        logging.debug("Model was purged, need to re-create")
        model = stable_whisper.load_faster_whisper(whisper_model, download_root=model_location, device=transcribe_device, cpu_threads=whisper_threads, num_workers=concurrent_transcriptions, compute_type=compute_type)

def delete_model():
    if clear_vram_on_complete and len(files_to_transcribe) == 0:
        global model
        logging.debug("Queue is empty, clearing/releasing VRAM")
        model = None
        gc.collect()

def isAudioFileExtension(file_extension):
    return file_extension.casefold() in \
        [ '.mp3', '.flac', '.wav', '.alac', '.ape', '.ogg', '.wma', '.m4a', '.m4b', '.aac', '.aiff' ]

def write_lrc(result, file_path):
    with open(file_path, "w") as file:
        for segment in result.segments:
            minutes, seconds = divmod(int(segment.start), 60)
            fraction = int((segment.start - int(segment.start)) * 100)
            file.write(f"[{minutes:02d}:{seconds:02d}.{fraction:02d}] {segment.text}\n")

def gen_subtitles(file_path: str, transcribe_or_translate: str, front=True, forceLanguage=None) -> None:
    """Generates subtitles for a video file.

    Args:
        file_path: str - The path to the video file.
        transcribe_or_translate: str - The type of transcription or translation to perform.
        front: bool - Whether to add the file to the front of the transcription queue. Default is True.
        forceLanguage: str - The language to force for transcription or translation. Default is None.
    """
    
    try:
        if not has_audio(file_path):
            logging.debug(f"{file_path} doesn't have any audio to transcribe!")
            return None
            
        if file_path not in files_to_transcribe:
            message = None
            if has_subtitle_language(file_path, skipifinternalsublang):
                message = f"{file_path} already has an internal subtitle we want, skipping generation"
            elif os.path.exists(file_path.rsplit('.', 1)[0] + subextension):
                message = f"{file_path} already has a subtitle created for this, skipping it"
            elif os.path.exists(file_path.rsplit('.', 1)[0] + subextensionSDH):
                message = f"{file_path} already has a SDH subtitle created for this, skipping it"
            if message != None:
                logging.info(message)
                return message
                
            if front:
                files_to_transcribe.insert(0, file_path)
            else:
                files_to_transcribe.append(file_path)
            logging.info(f"Added {os.path.basename(file_path)} for transcription.")
            # Start transcription for the file in a separate thread

            logging.info(f"{len(files_to_transcribe)} files in the queue for transcription")
            logging.info(f"Transcribing file: {os.path.basename(file_path)}")
            start_time = time.time()
            start_model()
            global force_detected_language_to
            if force_detected_language_to:
                forceLanguage = force_detected_language_to
                logging.info(f"Forcing language to {forceLanguage}")
            result = model.transcribe_stable(file_path, language=forceLanguage, task=transcribe_or_translate, progress_callback=progress, initial_prompt=custom_model_prompt)
            appendLine(result)
            file_name, file_extension = os.path.splitext(file_path)
            if isAudioFileExtension(file_extension) and lrc_for_audio_files:
                write_lrc(result, file_name + '.lrc')
            else:
                result.to_srt_vtt(file_name + subextension, word_level=word_level_highlight)
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            logging.info(f"Transcription of {os.path.basename(file_path)} is completed, it took {minutes} minutes and {seconds} seconds to complete.")
        else:
            logging.info(f"File {os.path.basename(file_path)} is already in the transcription list. Skipping.")

    except Exception as e:
        logging.info(f"Error processing or transcribing {file_path}: {e}")
    finally:
        if file_path in files_to_transcribe:
            files_to_transcribe.remove(file_path)
        delete_model()

def get_file_name_without_extension(file_path):
    file_name, file_extension = os.path.splitext(file_path)
    return file_name

def has_subtitle_language(video_file, target_language):
    try:
        with av.open(video_file) as container:
            subtitle_stream = next((stream for stream in container.streams if stream.type == 'subtitle' and 'language' in stream.metadata and stream.metadata['language'] == target_language), None)
            
            if subtitle_stream:
                logging.debug(f"Subtitles in '{target_language}' language found in the video.")
                return True
            else:
                logging.debug(f"No subtitles in '{target_language}' language found in the video.")
    except Exception as e:
        logging.info(f"An error occurred: {e}")
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

def refresh_plex_metadata(itemid: str, server_ip: str, plex_token: str) -> None:
    """
    Refreshes the metadata of a Plex library item.
    
    Args:
        itemid: The ID of the item in the Plex library whose metadata needs to be refreshed.
        server_ip: The IP address of the Plex server.
        plex_token: The Plex token used for authentication.
        
    Raises:
        Exception: If the server does not respond with a successful status code.
    """

    # Plex API endpoint to refresh metadata for a specific item
    url = f"{server_ip}/library/metadata/{itemid}/refresh"

    # Headers to include the Plex token for authentication
    headers = {
        "X-Plex-Token": plex_token,
    }

    # Sending the PUT request to refresh metadata
    response = requests.put(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        logging.info("Metadata refresh initiated successfully.")
    else:
        raise Exception(f"Error refreshing metadata: {response.status_code}")

def refresh_jellyfin_metadata(itemid: str, server_ip: str, jellyfin_token: str) -> None:
    """
    Refreshes the metadata of a Jellyfin library item.
    
    Args:
        itemid: The ID of the item in the Jellyfin library whose metadata needs to be refreshed.
        server_ip: The IP address of the Jellyfin server.
        jellyfin_token: The Jellyfin token used for authentication.
        
    Raises:
        Exception: If the server does not respond with a successful status code.
    """

    # Jellyfin API endpoint to refresh metadata for a specific item
    url = f"{server_ip}/Items/{itemid}/Refresh"

    # Headers to include the Jellyfin token for authentication
    headers = {
        "Authorization": f"MediaBrowser Token={jellyfin_token}",
    }

    # Cheap way to get the admin user id, and save it for later use.
    users = json.loads(requests.get(f"{server_ip}/Users", headers=headers).content)
    jellyfin_admin = get_jellyfin_admin(users)

    response = requests.get(f"{server_ip}/Users/{jellyfin_admin}/Items/{itemid}/Refresh", headers=headers)

    # Sending the PUT request to refresh metadata
    response = requests.post(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 204:
        logging.info("Metadata refresh queued successfully.")
    else:
        raise Exception(f"Error refreshing metadata: {response.status_code}")


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

def has_audio(file_path):
    try:
        with av.open(file_path) as container:
            return any(stream.type == 'audio' for stream in container.streams)
    except (av.AVError, UnicodeDecodeError):
        return False

def path_mapping(fullpath):
    if use_path_mapping:
        logging.debug("Updated path: " + fullpath.replace(path_mapping_from, path_mapping_to))
        return fullpath.replace(path_mapping_from, path_mapping_to)
    return fullpath

if monitor:
    # Define a handler class that will process new files
    class NewFileHandler(FileSystemEventHandler):
        def create_subtitle(self, event):
            # Only process if it's a file
            if not event.is_directory:
                file_path = event.src_path
                if has_audio(file_path):
                # Call the gen_subtitles function
                    logging.info(f"File: {path_mapping(file_path)} was added")
                    gen_subtitles(path_mapping(file_path), transcribe_or_translate, False)
        def on_created(self, event):
            self.create_subtitle(event)
        def on_modified(self, event):
            self.create_subtitle(event)

def transcribe_existing(transcribe_folders, forceLanguage=None):
    transcribe_folders = transcribe_folders.split("|")
    logging.info("Starting to search folders to see if we need to create subtitles.")
    logging.debug("The folders are:")
    for path in transcribe_folders:
        logging.debug(path)
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                gen_subtitles(path_mapping(file_path), transcribe_or_translate, False, forceLanguage)
    # if the path specified was actually a single file and not a folder, process it
    if os.path.isfile(path):
        if has_audio(path):
            gen_subtitles(path_mapping(path), transcribe_or_translate, False, forceLanguage) 
     # Set up the observer to watch for new files
    if monitor:
        observer = Observer()
        for path in transcribe_folders:
            if os.path.isdir(path):
                handler = NewFileHandler()
                observer.schedule(handler, path, recursive=True)
        observer.start()
        logging.info("Finished searching and queueing files for transcription. Now watching for new files.")

whisper_languages = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}

greetings_translations = {
    "en": "Hello, welcome to my lecture.",
    "zh": "你好，欢迎来到我的讲座。",
    "de": "Hallo, willkommen zu meiner Vorlesung.",
    "es": "Hola, bienvenido a mi conferencia.",
    "ru": "Привет, добро пожаловать на мою лекцию.",
    "ko": "안녕하세요, 제 강의에 오신 것을 환영합니다.",
    "fr": "Bonjour, bienvenue à mon cours.",
    "ja": "こんにちは、私の講義へようこそ。",
    "pt": "Olá, bem-vindo à minha palestra.",
    "tr": "Merhaba, dersime hoş geldiniz.",
    "pl": "Cześć, witaj na mojej wykładzie.",
    "ca": "Hola, benvingut a la meva conferència.",
    "nl": "Hallo, welkom bij mijn lezing.",
    "ar": "مرحبًا، مرحبًا بك في محاضرتي.",
    "sv": "Hej, välkommen till min föreläsning.",
    "it": "Ciao, benvenuto alla mia conferenza.",
    "id": "Halo, selamat datang di kuliah saya.",
    "hi": "नमस्ते, मेरे व्याख्यान में आपका स्वागत है।",
    "fi": "Hei, tervetuloa luentooni.",
    "vi": "Xin chào, chào mừng bạn đến với bài giảng của tôi.",
    "he": "שלום, ברוך הבא להרצאתי.",
    "uk": "Привіт, ласкаво просимо на мою лекцію.",
    "el": "Γεια σας, καλώς ήλθατε στη διάλεξή μου.",
    "ms": "Halo, selamat datang ke kuliah saya.",
    "cs": "Ahoj, vítejte na mé přednášce.",
    "ro": "Bună, bun venit la cursul meu.",
    "da": "Hej, velkommen til min forelæsning.",
    "hu": "Helló, üdvözöllek az előadásomon.",
    "ta": "வணக்கம், என் பாடத்திற்கு வரவேற்கிறேன்.",
    "no": "Hei, velkommen til foredraget mitt.",
    "th": "สวัสดีครับ ยินดีต้อนรับสู่การบรรยายของฉัน",
    "ur": "ہیلو، میری لیکچر میں خوش آمدید۔",
    "hr": "Pozdrav, dobrodošli na moje predavanje.",
    "bg": "Здравейте, добре дошли на моята лекция.",
    "lt": "Sveiki, sveiki atvykę į mano paskaitą.",
    "la": "Salve, gratias vobis pro eo quod meam lectionem excipitis.",
    "mi": "Kia ora, nau mai ki aku rorohiko.",
    "ml": "ഹലോ, എന്റെ പാഠത്തിലേക്ക് സ്വാഗതം.",
    "cy": "Helo, croeso i fy narlith.",
    "sk": "Ahoj, vitajte na mojej prednáške.",
    "te": "హలో, నా పాఠానికి స్వాగతం.",
    "fa": "سلام، خوش آمدید به سخنرانی من.",
    "lv": "Sveiki, laipni lūdzam uz manu lekciju.",
    "bn": "হ্যালো, আমার লেকচারে আপনাকে স্বাগতম।",
    "sr": "Здраво, добродошли на моје предавање.",
    "az": "Salam, mənim dərsimə xoş gəlmisiniz.",
    "sl": "Pozdravljeni, dobrodošli na moje predavanje.",
    "kn": "ಹಲೋ, ನನ್ನ ಭಾಷಣಕ್ಕೆ ಸುಸ್ವಾಗತ.",
    "et": "Tere, tere tulemast minu loengusse.",
    "mk": "Здраво, добредојдовте на мојата предавање.",
    "br": "Demat, kroget e oa d'an daol-labour.",
    "eu": "Kaixo, ongi etorri nire hitzaldi.",
    "is": "Halló, velkomin á fyrirlestur minn.",
    "hy": "Բարեւ, ողջույն եկավ իմ դասընթացի.",
    "ne": "नमस्ते, मेरो प्रवचनमा स्वागत छ।",
    "mn": "Сайн байна уу, миний хичээлд тавтай морилно уу.",
    "bs": "Zdravo, dobrodošli na moje predavanje.",
    "kk": "Сәлеметсіз бе, оқу сабағыма қош келдіңіз.",
    "sq": "Përshëndetje, mirësevini në ligjëratën time.",
    "sw": "Habari, karibu kwenye hotuba yangu.",
    "gl": "Ola, benvido á miña conferencia.",
    "mr": "नमस्कार, माझ्या व्याख्यानात आपले स्वागत आहे.",
    "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਮੇਰੀ ਵਾਰਤਾ ਵਿੱਚ ਤੁਹਾਨੂੰ ਜੀ ਆਇਆ ਨੂੰ ਸੁਆਗਤ ਹੈ।",
    "si": "හෙලෝ, මගේ වාර්තාවට ඔබේ ස්වාදයට සාමාජිකත්වයක්.",
    "km": "សួស្តី, សូមស្វាគមន៍មកកាន់អារម្មណ៍របស់ខ្ញុំ។",
    "sn": "Mhoro, wakaribisha kumusoro wangu.",
    "yo": "Bawo, ku isoro si wa orin mi.",
    "so": "Soo dhawoow, soo dhawoow marka laga hadlo kulambanayaashaaga.",
    "af": "Hallo, welkom by my lesing.",
    "oc": "Bonjorn, benvenguda a ma conferéncia.",
    "ka": "გამარჯობა, მესწარმეტყველება ჩემი ლექციაზე.",
    "be": "Прывітанне, запрашаем на маю лекцыю.",
    "tg": "Салом, ба лаҳзаи мавзӯъати ман хуш омадед.",
    "sd": "هيلو، ميري ليڪڪي ۾ خوش آيو.",
    "gu": "નમસ્તે, મારી પાઠશાળામાં આપનું સ્વાગત છે.",
    "am": "ሰላም፣ ለአንድነት የተመረጠን ትምህርት በመሆን እናመሰግናለን።",
    "yi": "העלאָ, ווילקומן צו מיין לעקטשער.",
    "lo": "ສະບາຍດີ, ຍິນດີນາງຂອງຂ້ອຍໄດ້ຍິນດີ.",
    "uz": "Salom, darsimda xush kelibsiz.",
    "fo": "Halló, vælkomin til mína fyrilestrar.",
    "ht": "Bonjou, byenveni nan leson mwen.",
    "ps": "سلام، مې لومړۍ کې خوش آمدید.",
    "tk": "Salam, dersimiňe hoş geldiňiz.",
    "nn": "Hei, velkomen til førelesinga mi.",
    "mt": "Hello, merħba għall-lezzjoni tiegħi.",
    "sa": "नमस्ते, मम उपन्यासे स्वागतं.",
    "lb": "Hallo, wëllkomm zu menger Lektioun.",
    "my": "မင်္ဂလာပါ၊ ကျေးဇူးတင်သည့်ကိစ္စသည်။",
    "bo": "བཀྲ་ཤིས་བདེ་ལེགས་འབད་བཅོས། ངའི་འཛིན་གྱི་སློབ་མའི་མིང་གི་འཕྲོད།",
    "tl": "Kamusta, maligayang pagdating sa aking leksyon.",
    "mg": "Manao ahoana, tonga soa sy tonga soa eto amin'ny lesona.",
    "as": "নমস্কাৰ, মোৰ পাঠলৈ আপোনাক স্বাগতম।",
    "tt": "Сәлам, лекциямга рәхмәт киләсез.",
    "haw": "Aloha, welina me ke kipa ana i ko'u ha'i 'ōlelo.",
    "ln": "Mbote, tango na zongisa mwa kilela yandi.",
    "ha": "Sannu, ka ci gaba da tattalin arziki na.",
    "ba": "Сәләм, лекцияғыма ҡуш тиңләгәнһүҙ.",
    "jw": "Halo, sugeng datang marang kulawargané.",
    "su": "Wilujeng, hatur nuhun ka lékturing abdi.",
}

if __name__ == "__main__":
    import uvicorn
    logging.info(f"Subgen v{subgen_version}")
    logging.info("Starting Subgen with listening webhooks!")
    logging.info(f"Transcriptions are limited to running {str(concurrent_transcriptions)} at a time")
    logging.info(f"Running {str(whisper_threads)} threads per transcription")
    logging.info(f"Using {transcribe_device} to encode")
    logging.info(f"Using faster-whisper")
    if transcribe_folders:
        transcribe_existing(transcribe_folders)
    uvicorn.run("subgen:app", host="0.0.0.0", port=int(webhookport), reload=reload_script_on_change, use_colors=True)
