subgen_version = '2024.3.5.188'

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
import requests
import av
import ffmpeg
import whisper

def convert_to_bool(in_bool):
    if isinstance(in_bool, bool):
        return in_bool
    else:
        value = str(in_bool).lower()
        return value not in ('false', 'off', '0', 0)

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
transcribe_folders = os.getenv('TRANSCRIBE_FOLDERS', '')
transcribe_or_translate = os.getenv('TRANSCRIBE_OR_TRANSLATE', 'transcribe')
force_detected_language_to = os.getenv('FORCE_DETECTED_LANGUAGE_TO', '')
hf_transformers = convert_to_bool(os.getenv('HF_TRANSFORMERS', False))
hf_batch_size = int(os.getenv('HF_BATCH_SIZE', 24))
clear_vram_on_complete = convert_to_bool(os.getenv('CLEAR_VRAM_ON_COMPLETE', True))
compute_type = os.getenv('COMPUTE_TYPE', 'auto')
append = convert_to_bool(os.getenv('APPEND', False))
timestamps = convert_to_bool(os.getenv('TIMESTAMPS', True))

if transcribe_device == "gpu":
    transcribe_device = "cuda"

app = FastAPI()
model = None
files_to_transcribe = []
subextension =  f".subgen.{whisper_model.split('.')[0]}.{namesublang}.srt"
subextensionSDH =  f".subgen.{whisper_model.split('.')[0]}.{namesublang}.sdh.srt"

if timestamps:
	# Define a custom class that modifies sys.stdout
    class TimestampedStdout:
        def __init__(self, stdout):
            self.stdout = stdout
        def write(self, x):
	        # Append the timestamp to every printed line
            self.stdout.write(x.replace("\n", "\n[%s] " % str(datetime.now())))
        def flush(self):
	        # Flush the output
            self.stdout.flush()
            # Replace sys.stdout with the custom class
    sys.stdout = TimestampedStdout(sys.stdout)

if debug:
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
else:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

logging.getLogger("multipart").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)

TIME_OFFSET = 5

def appendLine(result):
    if append:
        lastSegment = result.segments[-1].copy()
        lastSegment.id += 1
        lastSegment.start += TIME_OFFSET
        lastSegment.end += TIME_OFFSET
        date_time_str = datetime.now().strftime("%d %b %Y - %H:%M:%S")
        lastSegment.text = f"Transcribed by whisperAI with faster-whisper ({whisper_model}) on {date_time_str}"
        lastSegment.words = []
        # lastSegment.words[0].word = lastSegment.text
        # lastSegment.words = lastSegment.words[:len(lastSegment.words)-1]
        result.segments.append(lastSegment)

@app.get("/plex")
@app.get("/webhook")
@app.get("/jellyfin")
@app.get("/asr")
@app.get("/emby")
@app.get("/detect-language")
@app.get("/")
def handle_get_request(request: Request):
    return "You accessed this request incorrectly via a GET request.  See https://github.com/McCloudS/subgen for proper configuration"

@app.get("/status")
def status():
    return {"version" : f"Subgen {subgen_version}, stable-ts {stable_whisper.__version__}, whisper {whisper.__version__}"}

@app.post("/webhook")
async def print_warning():
    print("*** This is the legacy webhook.  You need to update to webhook urls to end in plex, tautulli, emby, or jellyfin instead of webhook. ***")
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
            try:
                refresh_plex_metadata(plex_json['Metadata']['ratingKey'], plexserver, plextoken)
                logging.info(f"Metadata for item {plex_json['Metadata']['ratingKey']} refreshed successfully.")
            except Exception as e:
                logging.error(f"Failed to refresh metadata for item {plex_json['Metadata']['ratingKey']}: {e}")
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
            try:
                refresh_jellyfin_metadata(ItemId, jellyfinserver, jellyfintoken)
                logging.info(f"Metadata for item {ItemId} refreshed successfully.")
            except Exception as e:
                logging.error(f"Failed to refresh metadata for item {ItemId}: {e}")
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
    
@app.post("/batch")
def batch(
        directory: Union[str, None] = Query(default=None),
        forceLanguage: Union[str, None] = Query(default=None)
):
    transcribe_existing(directory, forceLanguage)
    
# idea and some code for asr and detect language from https://github.com/ahmetoner/whisper-asr-webservice
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
        print(f"Transcribing file from Bazarr/ASR webhook")
        result = None
        #give the 'process' a random name so mutliple Bazaar transcribes can operate at the same time.
        random_name = random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890", k=6)
        
        start_time = time.time()
        start_model()
        
        files_to_transcribe.insert(0, f"Bazarr-detect-langauge-{random_name}")
        if(hf_transformers):
            result = model.transcribe(np.frombuffer(audio_file.file.read(), np.int16).flatten().astype(np.float32) / 32768.0, task=task, input_sr=16000, language=language, batch_size=hf_batch_size)
        else:
            result = model.transcribe_stable(np.frombuffer(audio_file.file.read(), np.int16).flatten().astype(np.float32) / 32768.0, task=task, input_sr=16000, language=language)
        appendLine(result)
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        print(f"Bazarr transcription is completed, it took {minutes} minutes and {seconds} seconds to complete.")
    except Exception as e:
        print(f"Error processing or transcribing Bazarr {audio_file.filename}: {e}")
    finally:
        if f"Bazarr-detect-langauge-{random_name}" in files_to_transcribe:
            files_to_transcribe.remove(f"Bazarr-detect-langauge-{random_name}")
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

@app.post("/detect-language")
def detect_language(
        audio_file: UploadFile = File(...),
        #encode: bool = Query(default=True, description="Encode audio first through ffmpeg") # This is always false from Bazarr
):    
    try:
        #give the 'process' a random name so mutliple Bazaar transcribes can operate at the same time.
        result = None
        random_name = random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890", k=6)
        start_model()

        files_to_transcribe.insert(0, f"Bazarr-detect-langauge-{random_name}")
        if(hf_transformers):
            detected_lang_code = model.transcribe(whisper.pad_or_trim(np.frombuffer(audio_file.file.read(), np.int16).flatten().astype(np.float32) / 32768.0), input_sr=16000, batch_size=hf_batch_size).language
        else:
            detected_lang_code = model.transcribe_stable(whisper.pad_or_trim(np.frombuffer(audio_file.file.read(), np.int16).flatten().astype(np.float32) / 32768.0), input_sr=16000).language
            
    except Exception as e:
        print(f"Error processing or transcribing Bazarr {audio_file.filename}: {e}")
        
    finally:
        if f"Bazarr-detect-langauge-{random_name}" in files_to_transcribe:
            files_to_transcribe.remove(f"Bazarr-detect-langauge-{random_name}")
        delete_model()

        return {"detected_language": get_lang_pair(whisper_languages, detected_lang_code), "language_code": detected_lang_code}

def start_model():
    global model
    if model is None:
        logging.debug("Model was purged, need to re-create")
        if(hf_transformers):
            logging.debug("Using Hugging Face Transformers, whisper_threads, concurrent_transcriptions, and model_location variables are ignored!")
            model = stable_whisper.load_hf_whisper(whisper_model, device=transcribe_device)
        else:
            model = stable_whisper.load_faster_whisper(whisper_model, download_root=model_location, device=transcribe_device, cpu_threads=whisper_threads, num_workers=concurrent_transcriptions, compute_type=compute_type)

def delete_model():
    if clear_vram_on_complete:
        if len(files_to_transcribe) == 0:
            global model
            logging.debug("Queue is empty, clearing/releasing VRAM")
            model = None
            gc.collect()

def get_lang_pair(whisper_languages, key):
  """Returns the other side of the pair in the Whisper languages dictionary.

  Args:
    whisper_languages: A dictionary of Whisper languages.
    key: The key to look up in the dictionary.

  Returns:
    The other side of the pair in the Whisper languages dictionary, or None if the
    key is not found in the dictionary.
  """

  other_side = whisper_languages.get(key)
  if other_side is None:
    return key
  else:
    return whisper_languages[other_side]

def gen_subtitles(file_path: str, transcribe_or_translate: str, front=True, forceLanguage=None) -> None:
    """Generates subtitles for a video file.

    Args:
        file_path: The path to the video file.
        transcription_or_translation: The type of transcription or translation to perform.
        front: Whether to add the file to the front of the transcription queue.
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
                print(message)
                return message
                
            if front:
                files_to_transcribe.insert(0, file_path)
            else:
                files_to_transcribe.append(file_path)
            print(f"Added {os.path.basename(file_path)} for transcription.")
            # Start transcription for the file in a separate thread

            print(f"{len(files_to_transcribe)} files in the queue for transcription")
            print(f"Transcribing file: {os.path.basename(file_path)}")
            start_time = time.time()
            start_model()
            global force_detected_language_to
            if force_detected_language_to:
                forceLanguage = force_detected_language_to
                print(f"Forcing language to {forceLanguage}")
            if(hf_transformers):
                result = model.transcribe(file_path, language=forceLanguage, batch_size=hf_batch_size, task=transcribe_or_translate)
            else:
                result = model.transcribe_stable(file_path, language=forceLanguage, task=transcribe_or_translate)
            appendLine(result)
            result.to_srt_vtt(get_file_name_without_extension(file_path) + subextension, word_level=word_level_highlight)
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            print(f"Transcription of {os.path.basename(file_path)} is completed, it took {minutes} minutes and {seconds} seconds to complete.")
        else:
            print(f"File {os.path.basename(file_path)} is already in the transcription list. Skipping.")

    except Exception as e:
        print(f"Error processing or transcribing {file_path}: {e}")
    finally:
        if file_path in files_to_transcribe:
            files_to_transcribe.remove(file_path)
        delete_model()

def get_file_name_without_extension(file_path):
    file_name, file_extension = os.path.splitext(file_path)
    return file_name

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
        print("Metadata refresh initiated successfully.")
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
    url = f"{server_ip}/library/metadata/{itemid}/refresh"

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
        print("Metadata refresh queued successfully.")
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
        container = av.open(file_path)
        for stream in container.streams:
            if stream.type == 'audio':
                return True
        return False
    except (av.AVError, UnicodeDecodeError):
        return False

def path_mapping(fullpath):
    if use_path_mapping:
        logging.debug("Updated path: " + fullpath.replace(path_mapping_from, path_mapping_to))
        return fullpath.replace(path_mapping_from, path_mapping_to)
    return fullpath

def transcribe_existing(transcribe_folders, forceLanguage=None):
    transcribe_folders = transcribe_folders.split("|")
    print("Starting to search folders to see if we need to create subtitles.")
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
                    
    print("Finished searching and queueing files for transcription")

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

if __name__ == "__main__":
    import uvicorn
    print(f"Subgen v{subgen_version}")
    print("Starting Subgen with listening webhooks!")
    print(f"Transcriptions are limited to running {str(concurrent_transcriptions)} at a time")
    print(f"Running {str(whisper_threads)} threads per transcription")
    print(f"Using {transcribe_device} to encode")
    if hf_transformers:
        print(f"Using Hugging Face Transformers")
    else:
        print(f"Using faster-whisper")
    if transcribe_folders:
        transcribe_existing(transcribe_folders)
    uvicorn.run("subgen:app", host="0.0.0.0", port=int(webhookport), reload=debug, use_colors=True)
