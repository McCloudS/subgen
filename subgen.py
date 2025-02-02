subgen_version = '2025.01.03'

from language_code import LanguageCode
from datetime import datetime
from threading import Lock
import os
import json
import xml.etree.ElementTree as ET
import threading
import sys
import time
import queue
import logging
import gc
import random
from typing import Union, Any, Optional
from fastapi import FastAPI, File, UploadFile, Query, Header, Body, Form, Request
from fastapi.responses import StreamingResponse
import numpy as np
import stable_whisper
from stable_whisper import Segment
import requests
import av
import ffmpeg
import whisper
import ast
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler
import faster_whisper
from io import BytesIO
import io

def get_key_by_value(d, value):
    reverse_dict = {v: k for k, v in d.items()}
    return reverse_dict.get(value)

def convert_to_bool(in_bool):
    # Convert the input to string and lower case, then check against true values
    return str(in_bool).lower() in ('true', 'on', '1', 'y', 'yes')
    
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
namesublang = os.getenv('NAMESUBLANG', '')
webhookport = int(os.getenv('WEBHOOKPORT', 9000))
word_level_highlight = convert_to_bool(os.getenv('WORD_LEVEL_HIGHLIGHT', False))
debug = convert_to_bool(os.getenv('DEBUG', True))
use_path_mapping = convert_to_bool(os.getenv('USE_PATH_MAPPING', False))
path_mapping_from = os.getenv('PATH_MAPPING_FROM', r'/tv')
path_mapping_to = os.getenv('PATH_MAPPING_TO', r'/Volumes/TV')
model_location = os.getenv('MODEL_PATH', './models')
monitor = convert_to_bool(os.getenv('MONITOR', False))
transcribe_folders = os.getenv('TRANSCRIBE_FOLDERS', '')
transcribe_or_translate = os.getenv('TRANSCRIBE_OR_TRANSLATE', 'transcribe').lower()
clear_vram_on_complete = convert_to_bool(os.getenv('CLEAR_VRAM_ON_COMPLETE', True))
compute_type = os.getenv('COMPUTE_TYPE', 'auto')
append = convert_to_bool(os.getenv('APPEND', False))
reload_script_on_change = convert_to_bool(os.getenv('RELOAD_SCRIPT_ON_CHANGE', False))
lrc_for_audio_files = convert_to_bool(os.getenv('LRC_FOR_AUDIO_FILES', True))
custom_regroup = os.getenv('CUSTOM_REGROUP', 'cm_sl=84_sl=42++++++1')
detect_language_length = int(os.getenv('DETECT_LANGUAGE_LENGTH', 30))
detect_language_offset = int(os.getenv('DETECT_LANGUAGE_START_OFFSET', 0))
skipifexternalsub = convert_to_bool(os.getenv('SKIPIFEXTERNALSUB', False))
skip_if_to_transcribe_sub_already_exist = convert_to_bool(os.getenv('SKIP_IF_TO_TRANSCRIBE_SUB_ALREADY_EXIST', True))
skipifinternalsublang = LanguageCode.from_string(os.getenv('SKIPIFINTERNALSUBLANG', ''))
plex_queue_next_episode = convert_to_bool(os.getenv('PLEX_QUEUE_NEXT_EPISODE', False))
plex_queue_series = convert_to_bool(os.getenv('PLEX_QUEUE_SERIES', False))
skip_lang_codes_list = (
    [LanguageCode.from_string(code) for code in os.getenv("SKIP_LANG_CODES", "").split("|")]
        if os.getenv('SKIP_LANG_CODES')
    else []
)
force_detected_language_to = LanguageCode.from_string(os.getenv('FORCE_DETECTED_LANGUAGE_TO', ''))
preferred_audio_languages = ( 
    [LanguageCode.from_string(code) for code in os.getenv('PREFERRED_AUDIO_LANGUAGES', 'eng').split("|")]
    if os.getenv('PREFERRED_AUDIO_LANGUAGES')
    else []
) # in order of preferrence
limit_to_preferred_audio_languages = convert_to_bool(os.getenv('LIMIT_TO_PREFERRED_AUDIO_LANGUAGE', False)) #TODO: add support for this
skip_if_audio_track_is_in_list = (
    [LanguageCode.from_string(code) for code in os.getenv('SKIP_IF_AUDIO_TRACK_IS', '').split("|")]
    if os.getenv('SKIP_IF_AUDIO_TRACK_IS')
    else []
)
subtitle_language_naming_type = os.getenv('SUBTITLE_LANGUAGE_NAMING_TYPE', 'ISO_639_2_B')
only_skip_if_subgen_subtitle = convert_to_bool(os.getenv('ONLY_SKIP_IF_SUBGEN_SUBTITLE', False))
skip_unknown_language = convert_to_bool(os.getenv('SKIP_UNKNOWN_LANGUAGE', False))
skip_if_language_is_not_set_but_subtitles_exist = convert_to_bool(os.getenv('SKIP_IF_LANGUAGE_IS_NOT_SET_BUT_SUBTITLES_EXIST', False)) 
should_whiser_detect_audio_language = convert_to_bool(os.getenv('SHOULD_WHISPER_DETECT_AUDIO_LANGUAGE', False))

try:
    kwargs = ast.literal_eval(os.getenv('SUBGEN_KWARGS', '{}') or '{}')
except ValueError:
    kwargs = {}
    logging.info("kwargs (SUBGEN_KWARGS) is an invalid dictionary, defaulting to empty '{}'")
    
if transcribe_device == "gpu":
    transcribe_device = "cuda"
        

VIDEO_EXTENSIONS = (
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".mpg", ".mpeg", 
    ".3gp", ".ogv", ".vob", ".rm", ".rmvb", ".ts", ".m4v", ".f4v", ".svq3", 
    ".asf", ".m2ts", ".divx", ".xvid"
)

AUDIO_EXTENSIONS = (
    ".mp3", ".wav", ".aac", ".flac", ".ogg", ".wma", ".alac", ".m4a", ".opus", 
    ".aiff", ".aif", ".pcm", ".ra", ".ram", ".mid", ".midi", ".ape", ".wv", 
    ".amr", ".vox", ".tak", ".spx", '.m4b'
)


app = FastAPI()
model = None

in_docker = os.path.exists('/.dockerenv')
docker_status = "Docker" if in_docker else "Standalone"
last_print_time = None

class DeduplicatedQueue(queue.Queue):
    """Queue that prevents duplicates in both queued and in-progress tasks."""
    def __init__(self):
        super().__init__()
        self._queued = set()    # Tracks paths in the queue
        self._processing = set()  # Tracks paths being processed
        self._lock = Lock()     # Ensures thread safety

    def put(self, item, block=True, timeout=None):
        with self._lock:
            path = item["path"]
            if path not in self._queued and path not in self._processing:
                super().put(item, block, timeout)
                self._queued.add(path)

    def get(self, block=True, timeout=None):
        item = super().get(block, timeout)
        with self._lock:
            path = item["path"]
            self._queued.discard(path)  # Remove from queued set
            self._processing.add(path)  # Mark as in-progress
        return item

    def task_done(self):
        super().task_done()
        with self._lock:
            # Assumes task_done() is called after processing the item from get()
            # If your workers process multiple items per get(), adjust logic here
            if self.unfinished_tasks == 0:
                self._processing.clear()  # Reset when all tasks are done

#start queue
task_queue = DeduplicatedQueue()

def transcription_worker():
    while True:
        task = task_queue.get()
                
        if "type" in task and task["type"] == "detect_language":
            detect_language_task(task['path'])
        elif 'Bazarr-' in task['path']:
            logging.info(f"Task {task['path']} is being handled by ASR.")
        else:
            logging.info(f"Task {task['path']} is being handled by Subgen.") 
            gen_subtitles(task['path'], task['transcribe_or_translate'], task['force_language'])
            task_queue.task_done()
        # show queue
        logging.debug(f"There are {task_queue.qsize()} tasks left in the queue.")

for _ in range(concurrent_transcriptions):
    threading.Thread(target=transcription_worker, daemon=True).start()

# Define a filter class to hide common logging we don't want to see
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
            "Calling on_"
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
            logging.debug("Force Update...")

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
def handle_get_request(request: Request):
    return {"You accessed this request incorrectly via a GET request.  See https://github.com/McCloudS/subgen for proper configuration"}

@app.get("/")
def webui():
    return {"The webui for configuration was removed on 1 October 2024, please configure via environment variables or in your Docker settings."}

@app.get("/status")
def status():
    return {"version" : f"Subgen {subgen_version}, stable-ts {stable_whisper.__version__}, faster-whisper {faster_whisper.__version__} ({docker_status})"}

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

            gen_subtitles_queue(path_mapping(fullpath), transcribe_or_translate)
    else:
        return {
            "message": "This doesn't appear to be a properly configured Tautulli webhook, please review the instructions again!"}

    return ""


@app.post("/plex")
def receive_plex_webhook(
        user_agent: Union[str] = Header(None),
        payload: Union[str] = Form(),
):
    try:
        plex_json = json.loads(payload)
        logging.debug(f"Raw response: {payload}")

        if "PlexMediaServer" not in user_agent:
            return {"message": "This doesn't appear to be a properly configured Plex webhook, please review the instructions again"}

        event = plex_json["event"]
        logging.debug(f"Plex event detected is: {event}")

        if (event == "library.new" and procaddedmedia) or (event == "media.play" and procmediaonplay):
            fullpath = get_plex_file_name(plex_json['Metadata']['ratingKey'], plexserver, plextoken)
            logging.debug("Path of file: " + fullpath)

            
            gen_subtitles_queue(path_mapping(fullpath), transcribe_or_translate)
            refresh_plex_metadata(plex_json['Metadata']['ratingKey'], plexserver, plextoken)
            if plex_queue_next_episode:
                gen_subtitles_queue(path_mapping(get_plex_file_name(get_next_plex_episode(plex_json['Metadata']['ratingKey']), plexserver, plextoken)), transcribe_or_translate)
            if plex_queue_series:
                current_rating_key = plex_json['Metadata']['ratingKey']

                # Process all episodes in the series starting from the current episode
                while current_rating_key is not None:
                    try:
                        # Queue the current episode
                        file_path = path_mapping(get_plex_file_name(current_rating_key, plexserver, plextoken))
                        gen_subtitles_queue(path_mapping(get_plex_file_name(get_next_plex_episode(current_rating_key), plexserver, plextoken)), transcribe_or_translate)

                        # Get the next episode
                        current_rating_key = get_next_plex_episode(current_rating_key)

                    except Exception as e:
                        logging.error(f"Error processing episode with ratingKey {current_rating_key} or reached end of series: {e}")
                        current_rating_key = None  # Stop processing on error

                logging.info("All episodes in the series have been queued.")
    except Exception as e:
        logging.error(f"Failed to process Plex webhook: {e}")

    return ""


@app.post("/jellyfin")
def receive_jellyfin_webhook(
        user_agent: str = Header(None),
        NotificationType: str = Body(None),
        file: str = Body(None),
        ItemId: str = Body(None),
):
    if "Jellyfin-Server" in user_agent:
        logging.debug(f"Jellyfin event detected is: {NotificationType}")
        logging.debug(f"itemid is: {ItemId}")

        if (NotificationType == "ItemAdded" and procaddedmedia) or (NotificationType == "PlaybackStart" and procmediaonplay):
            fullpath = get_jellyfin_file_name(ItemId, jellyfinserver, jellyfintoken)
            logging.debug(f"Path of file: {fullpath}")

            gen_subtitles_queue(path_mapping(fullpath), transcribe_or_translate)
            try:
                refresh_jellyfin_metadata(ItemId, jellyfinserver, jellyfintoken)
                logging.info(f"Metadata for item {ItemId} refreshed successfully.")
            except Exception as e:
                logging.error(f"Failed to refresh metadata for item {ItemId}: {e}")
    else:
        return {
            "message": "This doesn't appear to be a properly configured Jellyfin webhook, please review the instructions again!"}

    return ""


@app.post("/emby")
def receive_emby_webhook(
        user_agent: Union[str, None] = Header(None),
        data: Union[str, None] = Form(None),
):
    logging.debug("Raw response: %s", data)

    if not data:
        return ""

    data_dict = json.loads(data)
    event = data_dict['Event']
    logging.debug("Emby event detected is: " + event)

    # Check if it's a notification test event
    if event == "system.notificationtest":
        logging.info("Emby test message received!")
        return {"message": "Notification test received successfully!"}

    if (event == "library.new" and procaddedmedia) or (event == "playback.start" and procmediaonplay):
        fullpath = data_dict['Item']['Path']
        logging.debug("Path of file: " + fullpath)
        gen_subtitles_queue(path_mapping(fullpath), transcribe_or_translate)

    return ""
    
@app.post("/batch")
def batch(
        directory: Union[str, None] = Query(default=None),
        forceLanguage: Union[str, None] = Query(default=None)
):
    transcribe_existing(directory, LanguageCode.from_string(forceLanguage))
    
# idea and some code for asr and detect language from https://github.com/ahmetoner/whisper-asr-webservice
@app.post("//asr")
@app.post("/asr")
async def asr(
    task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None),
    video_file: Union[str, None] = Query(default=None),
    initial_prompt: Union[str, None] = Query(default=None),  # Not used by Bazarr
    audio_file: UploadFile = File(...),
    encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),  # Not used by Bazarr/always False
    output: Union[str, None] = Query(default="srt", enum=["txt", "vtt", "srt", "tsv", "json"]),
    word_timestamps: bool = Query(default=False, description="Word-level timestamps"),  # Not used by Bazarr
):
    try:
        logging.info(f"Transcribing file '{video_file}' from Bazarr/ASR webhook" if video_file else "Transcribing file from Bazarr/ASR webhook")
        
        result = None
        random_name = ''.join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890", k=6))

        if force_detected_language_to:
            language = force_detected_language_to.to_iso_639_1()
            logging.info(f"ENV FORCE_DETECTED_LANGUAGE_TO is set: Forcing detected language to {force_detected_language_to}")

        start_time = time.time()
        start_model()

        task_id = {'path': f"Bazarr-asr-{random_name}"}
        task_queue.put(task_id)

        args = {}
        args['progress_callback'] = progress
        
        file_content = audio_file.file.read()

        if encode:
            args['audio'] = file_content
        else:
            args['audio'] = np.frombuffer(file_content, np.int16).flatten().astype(np.float32) / 32768.0
            args['input_sr'] = 16000

        if custom_regroup:
            args['regroup'] = custom_regroup

        args.update(kwargs)

        result = model.transcribe(task=task, language=language, **args)
        appendLine(result)

        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        logging.info(
            f"Transcription of '{video_file}' from Bazarr complete, it took {minutes} minutes and {seconds} seconds to complete." if video_file 
            else f"Transcription complete, it took {minutes} minutes and {seconds} seconds to complete.")
    
    except Exception as e:
        logging.error(
            f"Error processing or transcribing Bazarr file: {video_file} -- Exception: {e}" if video_file
            else f"Error processing or transcribing Bazarr file Exception: {e}"
        )
    
    finally:
        await audio_file.close()
        task_queue.task_done()
        delete_model()
    
    if result:
        return StreamingResponse(
            iter(result.to_srt_vtt(filepath=None, word_level=word_level_highlight)),
            media_type="text/plain",
            headers={
                'Source': 'Transcribed using stable-ts from Subgen!',
            }
        )
    else:
        return
@app.post("//detect-language")
@app.post("/detect-language")
async def detect_language(
        audio_file: UploadFile = File(...),
        encode: bool = Query(default=True, description="Encode audio first through ffmpeg"), # This is always false from Bazarr
        detect_lang_length: int = Query(default=detect_language_length, description="Detect language on X seconds of the file"),
        detect_lang_offset: int = Query(default=detect_language_offset, description="Start Detect language X seconds into the file")
):    
    
    if force_detected_language_to:
        logging.info(f"language is: {force_detected_language_to.to_name()}")
        logging.debug(f"Skipping detect language, we have forced it as {force_detected_language_to.to_name()}")
        return {
            "detected_language": force_detected_language_to.to_name(),
            "language_code": force_detected_language_to.to_iso_639_1()
        }
        
    global detect_language_length, detect_language_offset
    detected_language = LanguageCode.NONE
    language_code = 'und'
    if force_detected_language_to:
            logging.info(f"ENV FORCE_DETECTED_LANGUAGE_TO is set: Forcing detected language to {force_detected_language_to}\n Returning without detection")
            return {"detected_language": force_detected_language_to.to_name(), "language_code": force_detected_language_to.to_iso_639_1()}
            
    # Log custom detection time settings if modified
    if detect_lang_length != detect_language_length:
        logging.info(f"Detecting language on the first {detect_lang_length} seconds of the audio.")
        detect_language_length = detect_lang_length

    if detect_lang_offset != detect_language_offset:
        logging.info(f"Offsetting language detection by {detect_language_offset} seconds.")
        detect_language_offset = detect_lang_offset
        #audio_file = extract_audio_segment_to_memory(audio_file, detect_language_offset, detect_language_length)
    try:
        start_model()
        random_name = ''.join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890", k=6))
        
        task_id = { 'path': f"Bazarr-detect-language-{random_name}" }        
        task_queue.put(task_id)
        args = {}
        #sample_rate = next(stream.rate for stream in av.open(audio_file.file).streams if stream.type == 'audio')
        #logging.info(f"Sample rate is: {sample_rate}")
        audio_file.file.seek(0)
        args['progress_callback'] = progress
        
        if encode:
            args['audio'] = extract_audio_segment_to_memory(audio_file, detect_language_offset, detect_language_length).read()
            args['input_sr'] = 16000
        else:
            #args['audio'] = whisper.pad_or_trim(np.frombuffer(audio_file.file.read(), np.int16).flatten().astype(np.float32) / 32768.0, args['input_sr'] * int(detect_language_length))
            args['audio'] = await get_audio_chunk(audio_file, detect_lang_offset, detect_lang_length)
            args['input_sr'] = 16000

        args.update(kwargs)
        detected_language = LanguageCode.from_name(model.transcribe(**args).language)
        logging.debug(f"Detected language: {detected_language.to_name()}")
        # reverse lookup of language -> code, ex: "english" -> "en", "nynorsk" -> "nn", ...
        language_code = detected_language.to_iso_639_1()
        logging.debug(f"Language Code: {language_code}")

    except Exception as e:
        logging.info(f"Error processing or transcribing Bazarr {audio_file.filename}: {e}")
        
    finally:
        #await audio_file.close()
        task_queue.task_done()
        delete_model()

        return {"detected_language": detected_language.to_name(), "language_code": language_code}

async def get_audio_chunk(audio_file, offset=detect_language_offset, length=detect_language_length, sample_rate=16000, audio_format=np.int16):
    """
    Extract a chunk of audio from a file, starting at the given offset and of the given length.
    
    :param audio_file: The audio file (UploadFile or file-like object).
    :param offset: The offset in seconds to start the extraction.
    :param length: The length in seconds for the chunk to be extracted.
    :param sample_rate: The sample rate of the audio (default 16000).
    :param audio_format: The audio format to interpret (default int16, 2 bytes per sample).
    
    :return: A numpy array containing the extracted audio chunk.
    """
    
    # Number of bytes per sample (for int16, 2 bytes per sample)
    bytes_per_sample = np.dtype(audio_format).itemsize
    
    # Calculate the start byte based on offset and sample rate
    start_byte = offset * sample_rate * bytes_per_sample
    
    # Calculate the length in bytes based on the length in seconds
    length_in_bytes = length * sample_rate * bytes_per_sample
    
    # Seek to the start position (this assumes the audio_file is a file-like object)
    await audio_file.seek(start_byte)
    
    # Read the required chunk of audio (length_in_bytes)
    chunk = await audio_file.read(length_in_bytes)
    
    # Convert the chunk into a numpy array (normalized to float32)
    audio_data = np.frombuffer(chunk, dtype=audio_format).flatten().astype(np.float32) / 32768.0
    
    return audio_data

def detect_language_task(path):
    detected_language = LanguageCode.NONE
    language_code = 'und'
    global detect_language_length 

    logger.info(f"Detecting language of file: {path} on the first {detect_language_length} seconds of the file")

    try:
        start_model()

        audio_segment = extract_audio_segment_to_memory(path, detect_language_offset, int(detect_language_length)).read()
        

        detected_language = LanguageCode.from_name(model.transcribe(audio_segment).language)
        logging.debug(f"Detected language: {detected_language.to_name()}")
        # reverse lookup of language -> code, ex: "english" -> "en", "nynorsk" -> "nn", ...
        language_code = detected_language.to_iso_639_1()
        logging.debug(f"Language Code: {language_code}")

    except Exception as e:
        logging.info(f"Error detectign language of file with whisper: {e}")
        
    finally:
        task_queue.task_done()
        delete_model()
        # put task to transcribe this with the detected language
        task_id = { 'path': path, "transcribe_or_translate": transcribe_or_translate, 'force_language': detected_language }
        task_queue.put(task_id)
        
        #maybe modify the file to contain detected language so we won't trigger this again
        
        return

def extract_audio_segment_to_memory(input_file, start_time, duration):
    """
    Extract a segment of audio from input_file, starting at start_time for duration seconds.
    
    :param input_file: UploadFile object or path to the input audio file
    :param start_time: Start time in seconds (e.g., 60 for 1 minute)
    :param duration: Duration in seconds (e.g., 30 for 30 seconds)
    :return: BytesIO object containing the audio segment
    """
    try:
        if hasattr(input_file, 'file') and hasattr(input_file.file, 'read'):  # Handling UploadFile
            input_file.file.seek(0)  # Ensure the file pointer is at the beginning
            input_stream = 'pipe:0'
            input_kwargs = {'input': input_file.file.read()}
        elif isinstance(input_file, str):  # Handling local file path
            input_stream = input_file
            input_kwargs = {}
        else:
            raise ValueError("Invalid input: input_file must be a file path or an UploadFile object.")

        logging.info(f"Extracting audio from: {input_stream}, start_time: {start_time}, duration: {duration}")

        # Run FFmpeg to extract the desired segment
        out, _ = (
            ffmpeg
            .input(input_stream, ss=start_time, t=duration)  # Set start time and duration
            .output('pipe:1', format='wav', acodec='pcm_s16le', ar=16000)  # Output to pipe as WAV
            .run(capture_stdout=True, capture_stderr=True, **input_kwargs)
        )

        # Check if the output is empty or null
        if not out:
            raise ValueError("FFmpeg output is empty, possibly due to invalid input.")
        
        return io.BytesIO(out)  # Convert output to BytesIO for in-memory processing

    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error: {e.stderr.decode()}")
        return None
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return None

    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error: {e.stderr.decode()}")
        return None
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return None

def start_model():
    global model
    if model is None:
        logging.debug("Model was purged, need to re-create")
        model = stable_whisper.load_faster_whisper(whisper_model, download_root=model_location, device=transcribe_device, cpu_threads=whisper_threads, num_workers=concurrent_transcriptions, compute_type=compute_type)

def delete_model():
    if clear_vram_on_complete and task_queue.qsize() == 0:
        global model
        logging.debug("Queue is empty, clearing/releasing VRAM")
        model = None
    gc.collect()

def isAudioFileExtension(file_extension):
    return file_extension.casefold() in \
        AUDIO_EXTENSIONS

def write_lrc(result, file_path):
    with open(file_path, "w") as file:
        for segment in result.segments:
            minutes, seconds = divmod(int(segment.start), 60)
            fraction = int((segment.start - int(segment.start)) * 100)
            # remove embedded newlines in text, since some players ignore text after newlines
            text = segment.text[:].replace('\n', '')
            file.write(f"[{minutes:02d}:{seconds:02d}.{fraction:02d}]{text}\n")

def gen_subtitles(file_path: str, transcription_type: str, force_language : LanguageCode = LanguageCode.NONE) -> None:
    """Generates subtitles for a video file.

    Args:
        file_path: str - The path to the video file.
        transcription_type: str - The type of transcription or translation to perform.
        force_language: str - The language to force for transcription or translation. Default is None.
    """

    try:
        logging.info(f"Added {os.path.basename(file_path)} for transcription.")
        logging.info(f"Transcribing file: {os.path.basename(file_path)}")
        logging.info(f"Transcribing file language: {force_language}")

        start_time = time.time()
        start_model()
        
        # Check if the file is an audio file before trying to extract audio 
        file_name, file_extension = os.path.splitext(file_path)
        is_audio_file = isAudioFileExtension(file_extension)
        
        data = file_path
        # Extract audio from the file if it has multiple audio tracks
        extracted_audio_file = handle_multiple_audio_tracks(file_path, force_language)
        if extracted_audio_file:
            data = extracted_audio_file.read()
        
        args = {}
        args['progress_callback'] = progress
            
        if custom_regroup:
            args['regroup'] = custom_regroup
            
        args.update(kwargs)
        
        result = model.transcribe(data, language=force_language.to_iso_639_1(), task=transcription_type, **args)

        appendLine(result)

        # If it is an audio file, write the LRC file
        if is_audio_file and lrc_for_audio_files:
            write_lrc(result, file_name + '.lrc')
        else:
            if not force_language:
                force_language = LanguageCode.from_string(result.language)
            result.to_srt_vtt(name_subtitle(file_path, force_language), word_level=word_level_highlight)

        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        logging.info(
            f"Transcription of {os.path.basename(file_path)} is completed, it took {minutes} minutes and {seconds} seconds to complete.")

    except Exception as e:
        logging.info(f"Error processing or transcribing {file_path} in {force_language}: {e}")

    finally:
        delete_model()
        
def define_subtitle_language_naming(language: LanguageCode, type):
    """
    Determines the naming format for a subtitle language based on the given type.

    Args:
        language (LanguageCode): The language code object containing methods to get different formats of the language name.
        type (str): The type of naming format desired, such as 'ISO_639_1', 'ISO_639_2_T', 'ISO_639_2_B', 'NAME', or 'NATIVE'.

    Returns:
        str: The language name in the specified format. If an invalid type is provided, it defaults to the language's name.
    """
    if namesublang:
        return namesublang
        # If we are translating, then we ALWAYS output an english file.
    switch_dict = {
        "ISO_639_1": language.to_iso_639_1,
        "ISO_639_2_T": language.to_iso_639_2_t,
        "ISO_639_2_B": language.to_iso_639_2_b,
        "NAME": language.to_name,
        "NATIVE": lambda : language.to_name(in_english=False)
    }
    if transcribe_or_translate == 'translate':
        language = LanguageCode.from_string('eng')
    return switch_dict.get(type, language.to_name)()

def name_subtitle(file_path: str, language: LanguageCode) -> str:
    """
    Name the the subtitle file to be written, based on the source file and the language of the subtitle.
    
    Args:
        file_path: The path to the source file.
        language: The language of the subtitle.
    
    Returns:
        The name of the subtitle file to be written.
    """
    return f"{os.path.splitext(file_path)[0]}.subgen.{whisper_model.split('.')[0]}.{define_subtitle_language_naming(language, subtitle_language_naming_type)}.srt"
        
def handle_multiple_audio_tracks(file_path: str, language: LanguageCode | None = None) -> BytesIO | None:
    """
    Handles the possibility of a media file having multiple audio tracks.
    
    If the media file has multiple audio tracks, it will extract the audio track of the selected language. Otherwise, it will extract the first audio track.
    
    Parameters:
    file_path (str): The path to the media file.
    language (LanguageCode | None): The language of the audio track to search for. If None, it will extract the first audio track.
    
    Returns:
    io.BytesIO  | None: The audio or None if no audio track was extracted.
    """
    audio_bytes = None
    audio_tracks = get_audio_tracks(file_path)

    if len(audio_tracks) > 1:
        logging.debug(f"Handling multiple audio tracks from {file_path} and planning to extract audio track of language {language}")
        logging.debug(
            "Audio tracks:\n"
            + "\n".join([f"  - {track['index']}: {track['codec']} {track['language']} {('default' if track['default'] else '')}" for track in audio_tracks])
        )

        if language is not None:
            audio_track = get_audio_track_by_language(audio_tracks, language)
        if audio_track is None:
            audio_track = audio_tracks[0]
        
        audio_bytes = extract_audio_track_to_memory(file_path, audio_track["index"])
        if audio_bytes is None:
            logging.error(f"Failed to extract audio track {audio_track['index']} from {file_path}")
            return None
    return audio_bytes

def extract_audio_track_to_memory(input_video_path, track_index) -> BytesIO | None:
    """
    Extract a specific audio track from a video file to memory using FFmpeg.

    Args:
        input_video_path (str): The path to the video file.
        track_index (int): The index of the audio track to extract. If None, skip extraction.

    Returns:
        io.BytesIO | None: The audio data as a BytesIO object, or None if extraction failed.
    """
    if track_index is None:
        logging.warning(f"Skipping audio track extraction for {input_video_path} because track index is None")
        return None

    try:
        # Use FFmpeg to extract the specific audio track and output to memory
        out, _ = (
            ffmpeg.input(input_video_path)
            .output(
                "pipe:",  # Direct output to a pipe
                map=f"0:{track_index}",  # Select the specific audio track
                format="wav",             # Output format
                ac=1,                     # Mono audio (optional)
                ar=16000,                 # Sample rate 16 kHz (recommended for speech models)
                loglevel="quiet"
            )
            .run(capture_stdout=True, capture_stderr=True)  # Capture output in memory
        )
        # Return the audio data as a BytesIO object
        return BytesIO(out)

    except ffmpeg.Error as e:
        print("An error occurred:", e.stderr.decode())
        return None

def get_audio_track_by_language(audio_tracks, language):
    """
    Returns the first audio track with the given language.
    
    Args:
        audio_tracks (list): A list of dictionaries containing information about each audio track.
        language (str): The language of the audio track to search for.
    
    Returns:
        dict: The first audio track with the given language, or None if no match is found.
    """
    for track in audio_tracks:
        if track['language'] == language:
            return track
    return None

def choose_transcribe_language(file_path, forced_language):
    """
    Determines the language to be used for transcription based on the provided
    file path and language preferences.

    Args:
        file_path: The path to the file for which the audio tracks are analyzed.
        forced_language: The language to force for transcription if specified.

    Returns:
        The language code to be used for transcription. It prioritizes the
        `forced_language`, then the environment variable `force_detected_language_to`,
        then the preferred audio language if available, and finally the default
        language of the audio tracks. Returns None if no language preference is
        determined.
    """
    
    logger.debug(f"choose_transcribe_language({file_path}, {forced_language})")
    
    if forced_language:
        logger.debug(f"ENV FORCE_LANGUAGE is set: Forcing language to {forced_language}")   
        return forced_language

    if force_detected_language_to:
        logger.debug(f"ENV FORCE_DETECTED_LANGUAGE_TO is set: Forcing detected language to {force_detected_language_to}")
        return force_detected_language_to

    audio_tracks = get_audio_tracks(file_path)
    
    found_track_in_language = find_language_audio_track(audio_tracks, preferred_audio_languages)
    if found_track_in_language:
        language = found_track_in_language
        if language:
            logger.debug(f"Preferred language found: {language}")
            return language
    
    default_language = find_default_audio_track_language(audio_tracks)
    if default_language:
        logger.debug(f"Default language found: {default_language}")
        return default_language

    return LanguageCode.NONE 

    
def get_audio_tracks(video_file):
    """
    Extracts information about the audio tracks in a file.

    Returns:
        List of dictionaries with information about each audio track.
        Each dictionary has the following keys:
            index (int): The stream index of the audio track.
            codec (str): The name of the audio codec.
            channels (int): The number of audio channels.
            language (LanguageCode): The language of the audio track.
            title (str): The title of the audio track.
            default (bool): Whether the audio track is the default for the file.
            forced (bool): Whether the audio track is forced.
            original (bool): Whether the audio track is the original.
            commentary (bool): Whether the audio track is a commentary.

    Example:
        >>> get_audio_tracks("french_movie_with_english_dub.mp4")
        [
            {
                "index": 0,
                "codec": "dts",
                "channels": 6,
                "language": LanguageCode.FRENCH,
                "title": "French",
                "default": True,
                "forced": False,
                "original": True,
                "commentary": False
            },
            {
                "index": 1,
                "codec": "aac",
                "channels": 2,
                "language":  LanguageCode.ENGLISH,
                "title": "English",
                "default": False,
                "forced": False,
                "original": False,
                "commentary": False
            }
        ]

    Raises:
        ffmpeg.Error: If FFmpeg fails to probe the file.
    """
    try:
        # Probe the file to get audio stream metadata
        probe = ffmpeg.probe(video_file, select_streams='a')
        audio_streams = probe.get('streams', [])
        
        # Extract information for each audio track
        audio_tracks = []
        for stream in audio_streams:
            audio_track = {
                "index": int(stream.get("index", None)),
                "codec": stream.get("codec_name", "Unknown"),
                "channels": int(stream.get("channels", None)),
                "language": LanguageCode.from_iso_639_2(stream.get("tags", {}).get("language", "Unknown")),
                "title": stream.get("tags", {}).get("title", "None"),
                "default": stream.get("disposition", {}).get("default", 0) == 1,
                "forced": stream.get("disposition", {}).get("forced", 0) == 1,
                "original": stream.get("disposition", {}).get("original", 0) == 1,
                "commentary": "commentary" in stream.get("tags", {}).get("title", "").lower()
            }
            audio_tracks.append(audio_track)    
        return audio_tracks

    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error: {e.stderr}")
        return []
    except Exception as e:
        logging.error(f"An error occurred while reading audio track information: {str(e)}")
        return []

def find_language_audio_track(audio_tracks, find_languages):
    """
    Checks if an audio track with any of the given languages is present in the list of audio tracks.
    Returns the first language from `find_languages` that matches.
    
    Args:
        audio_tracks (list): A list of dictionaries containing information about each audio track.
        find_languages (list): A list  language codes to search for.
    
    Returns:
        str or None: The first language found from `find_languages`, or None if no match is found.
    """
    for language in find_languages:
        for track in audio_tracks:
            if track['language'] == language:
                return language
    return None
def find_default_audio_track_language(audio_tracks):    
    """
    Finds the language of the default audio track in the given list of audio tracks.

    Args:
        audio_tracks (list): A list of dictionaries containing information about each audio track.
            Must contain the key "default" which is a boolean indicating if the track is the default track.

    Returns:
        str: The ISO 639-2 code of the language of the default audio track, or None if no default track was found.
    """
    for track in audio_tracks:
        if track['default'] is True:
            return track['language']
    return None
    
    
def gen_subtitles_queue(file_path: str, transcription_type: str, force_language: LanguageCode = LanguageCode.NONE) -> None:
    global task_queue
    
    if not has_audio(file_path):
        logging.debug(f"{file_path} doesn't have any audio to transcribe!")
        return
    
    if transcription_type == 'translate':
        force_language = LanguageCode.from_string('eng')
        # if we are translating, then we must do it to english only
    else:
        force_language = choose_transcribe_language(file_path, force_language)
    
    # check if we would like to detect audio language in case of no audio language specified. Will return here again with specified language from whisper
    if not force_language and should_whiser_detect_audio_language:
        # make a detect language task
        task_id = { 'path': file_path, 'type': "detect_language" }
        task_queue.put(task_id)
        logging.info(f"task_queue.put(task_id)({file_path}, detect_language)")
        return
    
    
    if have_to_skip(file_path, force_language):
        logging.debug(f"{file_path} already has subtitles in {force_language}, skipping.")
        return
    
    task = {
        'path': file_path,
        'transcribe_or_translate': transcription_type,
        'force_language': force_language
    }
    task['force_language'] = force_language
    task_queue.put(task)
    logging.info(f"task_queue.put(task)({task['path']}, {task['transcribe_or_translate']}, {task['force_language']})")

def have_to_skip(file_path: str, transcribe_language: LanguageCode) -> bool:
    """
    Determines whether subtitle generation should be skipped for a given file.

    Args:
        file_path: The path to the file to check for existing subtitles.
        transcribe_language: The language intended for transcription.

    Returns:
        True if subtitle generation should be skipped; otherwise, False.
    """
    file_name, file_extension = os.path.splitext(file_path)
    if isAudioFileExtension(file_extension) and lrc_for_audio_files:
        lrc_file_path = file_name + '.lrc'
    
        # Check if the file already exists
        if os.path.exists(lrc_file_path):
            logging.debug(f"The file {lrc_file_path} already exists. Skipping LRC creation.")
            return True
            
    if skip_unknown_language and transcribe_language == LanguageCode.NONE:
        logging.debug(f"{file_path} has unknown language, skipping.")
        return True
    
    # Check if subtitles in the desired transcription language already exist
    if skip_if_to_transcribe_sub_already_exist and has_subtitle_language(file_path, transcribe_language):
        logging.debug(f"{file_path} already has subtitles in {transcribe_language}, skipping.")
        return True

    # Check if subtitles in the specified internal language(s) should skip processing
    if skipifinternalsublang and has_subtitle_language(file_path, skipifinternalsublang):
        logging.debug(f"{file_path} has internal subtitles matching skip condition, skipping.")
        return True

    # Check if external subtitles exist for the specified language
    # Probably not use LanguageCode for this, but just check with strings, to be able to skip with custom named languages. 
    if LanguageCode.is_valid_language(namesublang):
        if skipifexternalsub and has_subtitle_language(file_path, LanguageCode.from_string(namesublang)):
            logging.debug(f"{file_path} has external subtitles in {namesublang}, skipping.")
            return True

    # Skip if any language in the skip list is detected in existing subtitles
    existing_sub_langs = get_subtitle_languages(file_path)
    if any(lang in skip_lang_codes_list for lang in existing_sub_langs):
        logging.debug(f"Languages in skip list {skip_lang_codes_list} detected in {file_path}, skipping.")
        return True

    audio_langs = get_audio_languages(file_path)
    if preferred_audio_languages in audio_langs:
        logging.debug(f"Preferred audio language {preferred_audio_languages} detected in {file_path}.")
        # maybe not skip if subtitle exist in preferred audio language, but not in another preferred audio language if the file has multiple audio tracks matching the preferred audio languages
    else:
        if limit_to_preferred_audio_languages:
            logging.debug(f"Only non-preferred audio language detected in {file_path}, skipping.")
            return True
        if any(lang in skip_if_audio_track_is_in_list for lang in audio_langs):
            logging.debug(f"Audio language in skip list {skip_if_audio_track_is_in_list} detected in {file_path}, skipping.")
            return True

    # If none of the conditions matched, do not skip
    return False
    
def get_subtitle_languages(video_path):
    """
    Extract language codes from each audio stream in the video file using pyav.
    :param video_path: Path to the video file
    :return: List of language codes for each subtitle stream
    """
    languages = []

    # Open the video file
    with av.open(video_path) as container:
        # Iterate through each audio stream
        for stream in container.streams.subtitles:
            # Access the metadata for each audio stream
            lang_code = stream.metadata.get('language')
            if lang_code:
                languages.append(LanguageCode.from_iso_639_2(lang_code))
            else:
                # Append 'und' (undefined) if no language metadata is present
                languages.append(LanguageCode.NONE)
    
    return languages

def get_file_name_without_extension(file_path):
    file_name, file_extension = os.path.splitext(file_path)
    return file_name

def get_audio_languages(video_path):
    """
    Extract language codes from each audio stream in the video file.

    :param video_path: Path to the video file
    :return: List of language codes for each audio stream
    """
    audio_tracks = get_audio_tracks(video_path)
    return [track['language'] for track in audio_tracks]    

def has_subtitle_language(video_file, target_language: LanguageCode):
    """
    Determines if a subtitle file with the target language is available for a specified video file.

    This function checks both within the video file and in its associated folder for subtitles
    matching the specified language.

    Args:
        video_file: The path to the video file.
        target_language: The language of the subtitle file to search for.

    Returns:
        bool: True if a subtitle file with the target language is found, False otherwise.
    """
    return has_subtitle_language_in_file(video_file, target_language) or has_subtitle_of_language_in_folder(video_file, target_language)

def has_subtitle_language_in_file(video_file: str, target_language: Union[LanguageCode, None]):
    """
    Checks if a video file contains subtitles with a specific language.

    Args:
        video_file (str): The path to the video file.
        target_language (LanguageCode | None): The language of the subtitle file to search for.

    Returns:
        bool: True if a subtitle file with the target language is found, False otherwise.
    """
    try:
        with av.open(video_file) as container:
            # Create a list of subtitle streams with 'language' metadata
            subtitle_streams = [
                stream for stream in container.streams 
                if stream.type == 'subtitle' and 'language' in stream.metadata
            ]
            
            # Skip logic if target_language is None
            if target_language is None:
                if skip_if_language_is_not_set_but_subtitles_exist and subtitle_streams:
                    logging.debug("Language is not set, but internal subtitles exist.")
                    return True
                if only_skip_if_subgen_subtitle:
                    logging.debug("Skipping since only external subgen subtitles are considered.")
                    return False  # Skip if only looking for external subgen subtitles

            # Check if any subtitle stream matches the target language
            for stream in subtitle_streams:
                # Convert the subtitle stream's language to a LanguageCode instance and compare
                stream_language = LanguageCode.from_string(stream.metadata.get('language', '').lower())
                if stream_language == target_language:
                    logging.debug(f"Subtitles in '{target_language}' language found in the video.")
                    return True

            logging.debug(f"No subtitles in '{target_language}' language found in the video.")
            return False
        
    except Exception as e:
        logging.error(f"An error occurred while checking the file with pyav: {type(e).__name__}: {e}")
        return False

def has_subtitle_of_language_in_folder(video_file, target_language: LanguageCode, recursion = True):
    """Checks if the given folder has a subtitle file with the given language.

    Args:
        video_file: The path of the video file.
        target_language: The language of the subtitle file that we are looking for.
        recursion: If True, search in subfolders of the given folder. If False,
            only search in the given folder.

    Returns:
        True if a subtitle file with the given language is found in the folder,
            False otherwise.
    """
    subtitle_extensions = ['.srt', '.vtt', '.sub', '.ass', '.ssa', '.idx', '.sbv', '.pgs', '.ttml', '.lrc']
    
    # just get the name of the movie e.g. movie.2025.remastered
    video_file_stripped = os.path.splitext(os.path.split(video_file)[1])[0]
    folder_path = os.path.dirname(video_file)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.isfile(file_path):
            root, ext = os.path.splitext(file_name)
            if root.startswith(video_file_stripped) and ext.lower() in subtitle_extensions:
                parts = root[len(video_file_stripped):].lstrip(".").split(".")
                
                has_subgen = "subgen" in parts  # Checks if "subgen" is in parts
                
                #checking this first because e.g  LanguageCode.from_string("subgen") == LanguageCode.NONE is equal to True. Maybe handle this better with a check with a function like is language code. To check if part is a valid language before comparing it to target_language
                
                if target_language == LanguageCode.NONE:
                    if only_skip_if_subgen_subtitle:
                        if has_subgen:
                            logger.debug("Subtitles from subgen found in the folder. ")
                            return skip_if_language_is_not_set_but_subtitles_exist
                        else:
                            #might be other subtitles that have subgen in the name
                            continue
                    logger.debug("Subtitles exist in the folder. and only_skip_if_subgen_subtitle is False.")
                    return skip_if_language_is_not_set_but_subtitles_exist                 
                
                if any(LanguageCode.from_string(part) == target_language for part in parts):
                    # If the subtitle is found, return True
                    if only_skip_if_subgen_subtitle:
                        if has_subgen:
                            logger.debug(f"Subtitles from subgen in '{target_language}' language found in the folder.")
                            return True
                        else:
                            #might be other subtitles that have subgen in the name
                            continue
                    logger.debug(f"Subtitles in '{target_language}' language found in the folder.")
                    return True
        elif os.path.isdir(file_path) and recursion: 
            # Looking in the subfolders of the video for subtitles
            if has_subtitle_of_language_in_folder(os.path.join(file_path, os.path.split(video_file)[1]) , target_language, False):
                # If the language is found in the subfolders, return True
                return True
    # If the language is not found, return False
    return False

def get_next_plex_episode(current_episode_rating_key):
    """
    Get the next episode's ratingKey based on the current episode in Plex.

    Args:
        current_episode_rating_key (str): The ratingKey of the current episode.

    Returns:
        str: The ratingKey of the next episode, or None if it's the last episode.
    """
    try:
        # Get current episode's metadata to fetch parent (season) ratingKey
        url = f"{plexserver}/library/metadata/{current_episode_rating_key}"
        headers = {"X-Plex-Token": plextoken}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Parse XML response
        root = ET.fromstring(response.content)
        
        # Find the show ID
        grandparent_rating_key = root.find(".//Video").get("grandparentRatingKey")
        if grandparent_rating_key is None:
            logging.debug(f"Show not found for episode {current_episode_rating_key}")
            return None
        
        # Find the parent season ratingKey
        parent_rating_key = root.find(".//Video").get("parentRatingKey")
        if parent_rating_key is None:
            logging.debug(f"Parent season not found for episode {current_episode_rating_key}")
            return None
        
        #parent_rating_key = 99707
        
        # Get the list of seasons
        url = f"{plexserver}/library/metadata/{grandparent_rating_key}/children"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        seasons = ET.fromstring(response.content).findall(".//Directory[@type='season']")
            
        # Get the list of episodes in the parent season
        url = f"{plexserver}/library/metadata/{parent_rating_key}/children"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        #print(response.content)

        # Parse XML response for the list of episodes
        episodes = ET.fromstring(response.content).findall(".//Video")
        episodes_in_season = ET.fromstring(response.content).get('size')
        
        # Find the current episode index and get the next one
        current_episode_number = None
        current_season_number = None
        next_season_number = None
        for episode in episodes:
            rating_key_element = episode.get("ratingKey")
            if int(rating_key_element) == int(current_episode_rating_key):
                current_episode_number = episode.get("index")
                current_season_number = episode.get("parentIndex")
            if rating_key_element is None:
                logging.warning(f"ratingKey not found for episode at index")
                continue
                
        # Find next season if it exists
        for season in seasons:
            if int(season.get("index")) == int(current_season_number)+1:
                print(f"next season is: {episode.get('ratingKey')}")
                print(season.get("title"))
                next_season_number = season.get("ratingKey")
            #if current_episode_number == int(
            
        # Find next episode
        if int(current_episode_number) == int(episodes_in_season) and next_season_number is not None:
            logging.debug("At end of season, try to find next season and first episode.")
            url = f"{plexserver}/library/metadata/{next_season_number}/children"
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            episodes = ET.fromstring(response.content).findall(".//Video")
            current_episode_number = 0
        for episode in episodes:
            if int(episode.get("index")) == int(current_episode_number)+1:
                return episode.get("ratingKey")

        logging.debug(f"No next episode found for {get_plex_file_name(current_episode_rating_key, plexserver, plextoken)}, possibly end of season or series")
        return None

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from Plex: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None

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
        if not is_valid_path(file_path):
            return False
        
        if not (has_video_extension(file_path) or  has_audio_extension(file_path)):
            # logging.debug(f"{file_path} is an not a video or audio file, skipping processing. skipping processing")
            return False

        with av.open(file_path) as container:
            # Check for an audio stream and ensure it has a valid codec
            for stream in container.streams:
                if stream.type == 'audio':
                    # Check if the stream has a codec and if it is valid
                    if stream.codec_context and stream.codec_context.name != 'none':
                        return True
                    else:
                        logging.debug(f"Unsupported or missing codec for audio stream in {file_path}")
            return False

    except (av.FFmpegError, UnicodeDecodeError):
        logging.debug(f"Error processing file {file_path}")
        return False

def is_valid_path(file_path):
    # Check if the path is a file
    if not os.path.isfile(file_path):
        # If it's not a file, check if it's a directory
        if not os.path.isdir(file_path):
            logging.warning(f"{file_path} is neither a file nor a directory. Are your volumes correct?")
            return False
        else:
            logging.debug(f"{file_path} is a directory, skipping processing as a file.")
            return False
    else:
        return True    

def has_video_extension(file_name):
    file_extension = os.path.splitext(file_name)[1].lower()  # Get the file extension
    return file_extension in VIDEO_EXTENSIONS

def has_audio_extension(file_name):
    file_extension = os.path.splitext(file_name)[1].lower()  # Get the file extension
    return file_extension in AUDIO_EXTENSIONS


def path_mapping(fullpath):
    if use_path_mapping:
        logging.debug("Updated path: " + fullpath.replace(path_mapping_from, path_mapping_to))
        return fullpath.replace(path_mapping_from, path_mapping_to)
    return fullpath

def is_file_stable(file_path, wait_time=2, check_intervals=3):
    """Returns True if the file size is stable for a given number of checks."""
    if not os.path.exists(file_path):
        return False
    
    previous_size = -1
    for _ in range(check_intervals):
        try:
            current_size = os.path.getsize(file_path)
        except OSError:
            return False  # File might still be inaccessible

        if current_size == previous_size:
            return True  # File is stable
        previous_size = current_size
        time.sleep(wait_time)
    
    return False  # File is still changing

if monitor:
    # Define a handler class that will process new files
    class NewFileHandler(FileSystemEventHandler):
        def create_subtitle(self, event):
            # Only process if it's a file
            if not event.is_directory:
                file_path = event.src_path
                if has_audio(file_path):
                    logging.info(f"File: {path_mapping(file_path)} was added")
                    gen_subtitles_queue(path_mapping(file_path), transcribe_or_translate)

        def handle_event(self, event):
            """Wait for stability before processing the file."""
            file_path = event.src_path
            if is_file_stable(file_path):
                self.create_subtitle(event)

        def on_created(self, event):
            time.sleep(5)  # Extra buffer time for new files
            self.handle_event(event)

        def on_modified(self, event):
            self.handle_event(event)

def transcribe_existing(transcribe_folders, forceLanguage : LanguageCode | None = None):
    transcribe_folders = transcribe_folders.split("|")
    logging.info("Starting to search folders to see if we need to create subtitles.")
    logging.debug("The folders are:")
    for path in transcribe_folders:
        logging.debug(path)
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                gen_subtitles_queue(path_mapping(file_path), transcribe_or_translate, forceLanguage)
    # if the path specified was actually a single file and not a folder, process it
    if os.path.isfile(path):
        if has_audio(path):
            gen_subtitles_queue(path_mapping(path), transcribe_or_translate, forceLanguage) 
     # Set up the observer to watch for new files
    if monitor:
        observer = Observer()
        for path in transcribe_folders:
            if os.path.isdir(path):
                handler = NewFileHandler()
                observer.schedule(handler, path, recursive=True)
        observer.start()
        logging.info("Finished searching and queueing files for transcription. Now watching for new files.")


if __name__ == "__main__":
    import uvicorn
    logging.info(f"Subgen v{subgen_version}")
    logging.info("Starting Subgen with listening webhooks!")
    logging.info(f"Transcriptions are limited to running {str(concurrent_transcriptions)} at a time")
    logging.info(f"Running {str(whisper_threads)} threads per transcription")
    logging.info(f"Using {transcribe_device} to encode")
    logging.info(f"Using faster-whisper")
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    if transcribe_folders:
        transcribe_existing(transcribe_folders)
    uvicorn.run("__main__:app", host="0.0.0.0", port=int(webhookport), reload=reload_script_on_change, use_colors=True)
