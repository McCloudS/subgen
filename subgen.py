subgen_version = '2026.02.9'

"""
ENVIRONMENT VARIABLES DOCUMENTATION

This application supports both new standardized environment variable names and legacy names for backwards compatibility. The new names follow a consistent naming convention: 

STANDARDIZED NAMING CONVENTION:
- Use UPPERCASE with underscores for separation
- Group related variables with consistent prefixes: 
  * PLEX_* for Plex server integration
  * JELLYFIN_* for Jellyfin server integration
  * PROCESS_* for media processing triggers
  * SKIP_* for all skip conditions
  * SUBTITLE_* for subtitle-related settings
  * WHISPER_* for Whisper model settings
  * TRANSCRIBE_* for transcription settings

BACKWARDS COMPATIBILITY: 
Legacy environment variable names are still supported. If both new and old names are set,
the new standardized name takes precedence. 

NEW NAME → OLD NAME (for backwards compatibility):
- PLEX_TOKEN → PLEXTOKEN
- PLEX_SERVER → PLEXSERVER
- JELLYFIN_TOKEN → JELLYFINTOKEN
- JELLYFIN_SERVER → JELLYFINSERVER
- PROCESS_ADDED_MEDIA → PROCADDEDMEDIA
- PROCESS_MEDIA_ON_PLAY → PROCMEDIAONPLAY
- SUBTITLE_LANGUAGE_NAME → NAMESUBLANG
- WEBHOOK_PORT → WEBHOOKPORT
- SKIP_IF_EXTERNAL_SUBTITLES_EXIST → SKIPIFEXTERNALSUB
- SKIP_IF_TARGET_SUBTITLES_EXIST → SKIP_IF_TO_TRANSCRIBE_SUB_ALREADY_EXIST
- SKIP_IF_INTERNAL_SUBTITLES_LANGUAGE → SKIPIFINTERNALSUBLANG
- SKIP_SUBTITLE_LANGUAGES → SKIP_LANG_CODES
- SKIP_IF_AUDIO_LANGUAGES → SKIP_IF_AUDIO_TRACK_IS
- SKIP_ONLY_SUBGEN_SUBTITLES → ONLY_SKIP_IF_SUBGEN_SUBTITLE
- SKIP_IF_NO_LANGUAGE_BUT_SUBTITLES_EXIST → SKIP_IF_LANGUAGE_IS_NOT_SET_BUT_SUBTITLES_EXIST

MIGRATION GUIDE:
Users can gradually migrate to the new names. Both will work simultaneously during the
transition period. The old names may be deprecated in future versions. 
"""

from language_code import LanguageCode
from datetime import datetime
from threading import Lock, Event, Timer
import os
import json
import xml.etree.ElementTree as ET
import threading
import sys
import time
import queue
import logging
import gc
import hashlib
from typing import Union, Any, Optional
from fastapi import FastAPI, File, UploadFile, Query, Header, Body, Form, Request
from fastapi.responses import StreamingResponse
import numpy as np
import stable_whisper
from stable_whisper import Segment
import requests
import av
import ffmpeg
import ast
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler
import faster_whisper
from io import BytesIO
import io
import torch
import ctypes, ctypes.util
from typing import List

def convert_to_bool(in_bool):
    # Convert the input to string and lower case, then check against true values
    return str(in_bool).lower() in ('true', 'on', '1', 'y', 'yes')

def get_env_with_fallback(new_name: str, old_name: str, default_value=None, convert_func=None):
    """
    Get environment variable with backwards compatibility fallback.
    
    Args:
        new_name: The new standardized environment variable name
        old_name: The legacy environment variable name for backwards compatibility
        default_value: Default value if neither variable is set
        convert_func: Optional function to convert the value (e.g., convert_to_bool, int)
    
    Returns:
        The environment variable value, converted if convert_func is provided
    """
    # Try new name first, then fall back to old name
    value = os.getenv(new_name) or os.getenv(old_name)
    
    if value is None:
        value = default_value
    
    # Apply conversion function if provided
    if convert_func and value is not None:
        return convert_func(value)
    
    return value
    
# Server Integration - with backwards compatibility
plextoken = get_env_with_fallback('PLEX_TOKEN', 'PLEXTOKEN', 'token here')
plexserver = get_env_with_fallback('PLEX_SERVER', 'PLEXSERVER', 'http://192.168.1.111:32400')
jellyfintoken = get_env_with_fallback('JELLYFIN_TOKEN', 'JELLYFINTOKEN', 'token here')
jellyfinserver = get_env_with_fallback('JELLYFIN_SERVER', 'JELLYFINSERVER', 'http://192.168.1.111:8096')

# Whisper Configuration
whisper_model = os.getenv('WHISPER_MODEL', 'medium')
whisper_threads = int(os.getenv('WHISPER_THREADS', 4))
concurrent_transcriptions = int(os.getenv('CONCURRENT_TRANSCRIPTIONS', 2))
transcribe_device = os.getenv('TRANSCRIBE_DEVICE', 'cpu')

# Processing Control - with backwards compatibility
procaddedmedia = get_env_with_fallback('PROCESS_ADDED_MEDIA', 'PROCADDEDMEDIA', True, convert_to_bool)
procmediaonplay = get_env_with_fallback('PROCESS_MEDIA_ON_PLAY', 'PROCMEDIAONPLAY', True, convert_to_bool)

# Subtitle Configuration - with backwards compatibility
namesublang = get_env_with_fallback('SUBTITLE_LANGUAGE_NAME', 'NAMESUBLANG', '')

# System Configuration - with backwards compatibility
webhookport = get_env_with_fallback('WEBHOOK_PORT', 'WEBHOOKPORT', 9000, int)
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
detect_language_offset = int(os.getenv('DETECT_LANGUAGE_OFFSET', 0))
model_cleanup_delay = int(os.getenv('MODEL_CLEANUP_DELAY', 30))
asr_timeout = int(os.getenv('ASR_TIMEOUT', 18000))

# Skip Configuration - with backwards compatibility
skipifexternalsub = get_env_with_fallback('SKIP_IF_EXTERNAL_SUBTITLES_EXIST', 'SKIPIFEXTERNALSUB', False, convert_to_bool)
skip_if_to_transcribe_sub_already_exist = get_env_with_fallback('SKIP_IF_TARGET_SUBTITLES_EXIST', 'SKIP_IF_TO_TRANSCRIBE_SUB_ALREADY_EXIST', True, convert_to_bool)
skipifinternalsublang = LanguageCode.from_string(get_env_with_fallback('SKIP_IF_INTERNAL_SUBTITLES_LANGUAGE', 'SKIPIFINTERNALSUBLANG', ''))
plex_queue_next_episode = convert_to_bool(os.getenv('PLEX_QUEUE_NEXT_EPISODE', False))
plex_queue_season = convert_to_bool(os.getenv('PLEX_QUEUE_SEASON', False))
plex_queue_series = convert_to_bool(os.getenv('PLEX_QUEUE_SERIES', False))
# Language and Skip Configuration - with backwards compatibility
skip_lang_codes_list = (
    [LanguageCode.from_string(code) for code in get_env_with_fallback('SKIP_SUBTITLE_LANGUAGES', 'SKIP_LANG_CODES', '').split("|")]
        if get_env_with_fallback('SKIP_SUBTITLE_LANGUAGES', 'SKIP_LANG_CODES')
    else []
)
force_detected_language_to = LanguageCode.from_string(os.getenv('FORCE_DETECTED_LANGUAGE_TO', ''))
preferred_audio_languages = [
    LanguageCode.from_string(code) 
    for code in os.getenv('PREFERRED_AUDIO_LANGUAGES', 'eng').split("|")
] # in order of preference
limit_to_preferred_audio_languages = convert_to_bool(os.getenv('LIMIT_TO_PREFERRED_AUDIO_LANGUAGE', False)) #TODO: add support for this
skip_if_audio_track_is_in_list = (
    [LanguageCode.from_string(code) for code in get_env_with_fallback('SKIP_IF_AUDIO_LANGUAGES', 'SKIP_IF_AUDIO_TRACK_IS', '').split("|")]
    if get_env_with_fallback('SKIP_IF_AUDIO_LANGUAGES', 'SKIP_IF_AUDIO_TRACK_IS')
    else []
)

# Additional Subtitle Configuration - with backwards compatibility
subtitle_language_naming_type = os.getenv('SUBTITLE_LANGUAGE_NAMING_TYPE', 'ISO_639_2_B')
only_skip_if_subgen_subtitle = get_env_with_fallback('SKIP_ONLY_SUBGEN_SUBTITLES', 'ONLY_SKIP_IF_SUBGEN_SUBTITLE', False, convert_to_bool)
skip_unknown_language = convert_to_bool(os.getenv('SKIP_UNKNOWN_LANGUAGE', False))
skip_if_language_is_not_set_but_subtitles_exist = get_env_with_fallback('SKIP_IF_NO_LANGUAGE_BUT_SUBTITLES_EXIST', 'SKIP_IF_LANGUAGE_IS_NOT_SET_BUT_SUBTITLES_EXIST', False, convert_to_bool)
should_whiser_detect_audio_language = convert_to_bool(os.getenv('SHOULD_WHISPER_DETECT_AUDIO_LANGUAGE', False))
show_in_subname_subgen = convert_to_bool(os.getenv('SHOW_IN_SUBNAME_SUBGEN', True))
show_in_subname_model = convert_to_bool(os.getenv('SHOW_IN_SUBNAME_MODEL', True))

# Advanced Configuration
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
    ".amr", ".vox", ".tak", ".spx", ".m4b", ".mka"
)

app = FastAPI()
model = None
model_cleanup_timer = None
model_cleanup_lock = Lock()

in_docker = os.path.exists('/.dockerenv')
docker_status = "Docker" if in_docker else "Standalone"

# ============================================================================
# TASK RESULT STORAGE (for blocking endpoints)
# ============================================================================

class TaskResult:
    """Stores the result of a queued task for blocking retrieval"""
    def __init__(self):
        self.result = None
        self.error = None
        self.done = Event()
    
    def set_result(self, result):
        self.result = result
        self.done.set()
    
    def set_error(self, error):
        self.error = error
        self.done.set()
    
    def wait(self, timeout=None):
        """Block until result is ready. Returns True if completed, False if timeout."""
        return self.done.wait(timeout)

# Dictionary to store task results keyed by task_id
# MEMLEAK-FIX-1: Entries are cleaned up in /asr endpoint finally block to prevent unbounded growth
task_results = {}
task_results_lock = Lock()

# ============================================================================
# HASH GENERATION FOR DEDUPLICATION
# ============================================================================

def generate_audio_hash(audio_content: bytes, task: str = None, language: str = None) -> str:
    """
    Generate a deterministic hash from audio content and optional parameters. 
    
    Same audio + same task + same language = always same hash. 
    This ensures duplicate requests are caught by the queue. 
    
    Args:
        audio_content: Raw audio bytes from uploaded file
        task: Optional task type ('transcribe' or 'translate')
        language: Optional target language code
        
    Returns: 
        SHA256 hash (first 16 chars for brevity in logs)
    """
    hash_input = audio_content
    
    # Include task and language for fine-grained deduplication
    if task:
        hash_input += task.encode('utf-8')
    if language:
        hash_input += language.encode('utf-8')
    
    full_hash = hashlib.sha256(hash_input).hexdigest()
    return full_hash[:16] # Use first 16 chars for shorter IDs in logs

# ============================================================================
# REFACTORED DEDUPLICATED QUEUE WITH BETTER TRACKING
# ============================================================================

class DeduplicatedQueue(queue.PriorityQueue):
    """Queue that prevents duplicates, handles priority, and tracks status."""
    def __init__(self):
        super().__init__()
        self._queued = set()     # Tracks task IDs waiting in queue
        self._processing = set() # Tracks task IDs currently being handled
        self._lock = Lock()

    def put(self, item, block=True, timeout=None):
        with self._lock:
            task_id = item["path"]
            if task_id not in self._queued and task_id not in self._processing:
                # Priority: 0 (Detect), 1 (ASR), 2 (Transcribe)
                task_type = item.get("type", "transcribe")
                priority = 0 if task_type == "detect_language" else (1 if task_type == "asr" else 2)
                
                # PriorityQueue requires a tuple: (priority, tie_breaker, item)
                super().put((priority, time.time(), item), block, timeout)
                self._queued.add(task_id)
                return True
            return False

    def get(self, block=True, timeout=None):
        # PriorityQueue returns the tuple, we want just the item
        priority, timestamp, item = super().get(block, timeout)
        with self._lock:
            task_id = item["path"]
            self._queued.discard(task_id)
            self._processing.add(task_id)
        return item

    def mark_done(self, item):
        with self._lock:
            task_id = item["path"]
            self._processing.discard(task_id)

    def is_idle(self):
        with self._lock:
            return self.empty() and len(self._processing) == 0

    def is_active(self, task_id):
        """Checks if a task_id is currently queued or processing."""
        with self._lock:
            return task_id in self._queued or task_id in self._processing

    def get_queued_tasks(self):
        with self._lock:
            return list(self._queued)

    def get_processing_tasks(self):
        with self._lock:
            return list(self._processing)

# Start queue
task_queue = DeduplicatedQueue()

# ============================================================================
# TRANSCRIPTION WORKER
# ============================================================================

def transcription_worker():
    """Main worker thread with centralized logging and status tracking."""
    while True:
        task = None
        try:
            task = task_queue.get(block=True, timeout=1)
            task_type = task.get("type", "transcribe")
            path = task.get("path", "unknown")
            display_name = os.path.basename(path) if ("/" in str(path) or "\\" in str(path)) else path
            
            # Status for START log
            proc_count = len(task_queue.get_processing_tasks())
            queue_count = len(task_queue.get_queued_tasks())
            logging.info(f"WORKER START : [{task_type.upper():<10}] {display_name:^40} | Jobs: {proc_count} processing, {queue_count} queued")
            
            start_time = time.time()
            if task_type == "detect_language": 
                if "audio_content" in task: 
                    detect_language_from_upload(task)
                else: 
                    # Pass the full task data so we don't lose the Plex ID
                    detect_language_task(task['path'], original_task_data=task)
            elif task_type == "asr":
                asr_task_worker(task)
            else: # transcribe
                gen_subtitles(task['path'], task['transcribe_or_translate'], task['force_language'])
                
                # --- METADATA REFRESH LOGIC ---
                # This runs ONLY after subtitles are successfully generated
                if 'plex_item_id' in task:
                    try:
                        logging.info(f"Refreshing Plex Metadata for item {task['plex_item_id']}")
                        refresh_plex_metadata(task['plex_item_id'], task['plex_server'], task['plex_token'])
                    except Exception as e:
                        logging.error(f"Failed to refresh Plex metadata: {e}")
                
                if 'jellyfin_item_id' in task:
                    try:
                        logging.info(f"Refreshing Jellyfin Metadata for item {task['jellyfin_item_id']}")
                        refresh_jellyfin_metadata(task['jellyfin_item_id'], task['jellyfin_server'], task['jellyfin_token'])
                    except Exception as e:
                        logging.error(f"Failed to refresh Jellyfin metadata: {e}")
                # ------------------------------
            
            # Status for FINISH log
            elapsed = time.time() - start_time
            m, s = divmod(int(elapsed), 60)
            remaining_queued = len(task_queue.get_queued_tasks())
            logging.info(f"WORKER FINISH: [{task_type.upper():<10}] {display_name:^40} in {m}m {s}s | Remaining: {remaining_queued} queued")

        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Error processing task: {e}", exc_info=True)
        finally:
            if task:
                task_queue.task_done()
                task_queue.mark_done(task)
                delete_model()
# Create worker threads
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
else:
    level = logging.INFO

logging.basicConfig(
    stream=sys.stderr, 
    level=level, 
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"  # This removes the ,123 part
)

# Get the root logger
logger = logging.getLogger()
logger.setLevel(level) # Set the logger level

for handler in logger.handlers:
    handler.addFilter(MultiplePatternsFilter())

logging.getLogger("multipart").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


class ProgressHandler:
    def __init__(self, filename):
        self.filename = filename
        self.start_time = time.time()
        self.last_print_time = 0
        self.interval = 5 

    def __call__(self, seek, total):
        if docker_status == 'Docker' or debug:
            current_time = time.time()
            if self.last_print_time == 0 or (current_time - self.last_print_time) >= self.interval:
                self.last_print_time = current_time
                
                # 1. Math for Metrics
                pct = int((seek / total) * 100) if total > 0 else 0
                elapsed = current_time - self.start_time
                speed = seek / elapsed if elapsed > 0 else 0
                eta = (total - seek) / speed if speed > 0 else 0

                # 2. Precise Time Formatting (removes milliseconds)
                def fmt_t(seconds):
                    m, s = divmod(int(seconds), 60)
                    h, m = divmod(m, 60)
                    if h > 0:
                        return f"{h}:{m:02d}:{s:02d}"
                    return f"{m:02d}:{s:02d}"

                # 3. Get Queue Stats
                proc = len(task_queue.get_processing_tasks())
                queued = len(task_queue.get_queued_tasks())

                # 4. Alignment Logic
                # :<40  = Left-align, 40 chars wide (Filename)
                # :>3   = Right-align, 3 chars wide (Percentage)
                # :>5   = Right-align, 5 chars wide (Seconds)
                # :>5   = Right-align, 5 chars wide (Time strings)
                
                clean_name = (self.filename[:37] + '..') if len(self.filename) > 40 else self.filename
                
                logging.info(
                    f"[ {clean_name:<40}] {pct:>3}% | "
                    f"{int(seek):>5}/{int(total):<5}s "
                    f"[{fmt_t(elapsed):>5}<{fmt_t(eta):>5}, {speed:>5.2f}s/s] | "
                    f"Jobs: {proc} processing, {queued} queued"
                )
                
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
            words=[], # Empty list for words
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
    return {"You accessed this request incorrectly via a GET request. See https://github.com/McCloudS/subgen for proper configuration"}

@app.get("/")
def webui():
    return {"The webui for configuration was removed on 1 October 2024, please configure via environment variables or in your Docker settings. "}

@app.get("/status")
def status():
    return {"version": f"Subgen {subgen_version}, stable-ts {stable_whisper.__version__}, faster-whisper {faster_whisper.__version__} ({docker_status})"}

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
            logging.debug(f"Full file path: {fullpath}")

            gen_subtitles_queue(path_mapping(fullpath), transcribe_or_translate)
    else:
        return {
            "message": "This doesn't appear to be a properly configured Tautulli webhook, please review the instructions again!"}

    return ""

@app.post("/plex")
@app.post("/plex")
def receive_plex_webhook(
        user_agent: Union[str] = Header(None),
        payload: Union[str] = Form(),
):
    try:
        plex_json = json.loads(payload)
        #logging.debug(f"Raw response: {payload}")

        if "PlexMediaServer" not in user_agent:
            return {"message": "This doesn't appear to be a properly configured Plex webhook, please review the instructions again"}

        event = plex_json["event"]
        logging.debug(f"Plex event detected is: {event}")

        if (event == "library.new" and procaddedmedia) or (event == "media.play" and procmediaonplay):
            rating_key = plex_json['Metadata']['ratingKey']
            fullpath = get_plex_file_name(rating_key, plexserver, plextoken)
            logging.debug(f"Full file path: {fullpath}")

            # Queue the current item with its specific ID for refreshing
            gen_subtitles_queue(
                path_mapping(fullpath), 
                transcribe_or_translate, 
                plex_item_id=rating_key, 
                plex_server=plexserver, 
                plex_token=plextoken
            )
            
            # Note: refresh_plex_metadata is removed here; it is now handled by the worker thread.

            if plex_queue_next_episode:
                next_key = get_next_plex_episode(plex_json['Metadata']['ratingKey'], stay_in_season=False)
                if next_key:
                    next_file = get_plex_file_name(next_key, plexserver, plextoken)
                    gen_subtitles_queue(
                        path_mapping(next_file), 
                        transcribe_or_translate,
                        plex_item_id=next_key, # Pass the NEXT ID so it refreshes when done
                        plex_server=plexserver,
                        plex_token=plextoken
                    )

            if plex_queue_series or plex_queue_season:
                current_rating_key = plex_json['Metadata']['ratingKey']
                stay_in_season = plex_queue_season # Determine if we're staying in the season or not

                while current_rating_key is not None:
                    try:
                        # Queue the current episode
                        file_path = path_mapping(get_plex_file_name(current_rating_key, plexserver, plextoken))
                        
                        gen_subtitles_queue(
                            file_path, 
                            transcribe_or_translate,
                            plex_item_id=current_rating_key, # Pass the specific loop ID for refreshing
                            plex_server=plexserver,
                            plex_token=plextoken
                        )
                        
                        logging.debug(f"Queued episode with ratingKey {current_rating_key}")

                        # Get the next episode
                        next_episode_rating_key = get_next_plex_episode(current_rating_key, stay_in_season=stay_in_season)
                        if next_episode_rating_key is None:
                            break # Exit the loop if no next episode
                        current_rating_key = next_episode_rating_key

                    except Exception as e:
                        logging.error(f"Error processing episode with ratingKey {current_rating_key} or reached end of series: {e}")
                        break # Stop processing on error

                logging.info("All episodes in the series (or season) have been queued.")

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
            logging.debug(f"Full file path: {fullpath}")

            # Queue item with Jellyfin metadata ID for delayed refresh
            gen_subtitles_queue(
                path_mapping(fullpath), 
                transcribe_or_translate,
                jellyfin_item_id=ItemId,
                jellyfin_server=jellyfinserver,
                jellyfin_token=jellyfintoken
            )
            
            # Note: refresh_jellyfin_metadata removed here; handled by worker.
    else:
        return {
            "message": "This doesn't appear to be a properly configured Jellyfin webhook, please review the instructions again!"}

    return ""

@app.post("/emby")
def receive_emby_webhook(
        user_agent: Union[str, None] = Header(None),
        data: Union[str, None] = Form(None),
):
    #logging.debug("Raw response: %s", data)

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
        logging.debug(f"Full file path: {fullpath}")
        gen_subtitles_queue(path_mapping(fullpath), transcribe_or_translate)

    return ""
    
@app.post("/batch")
def batch(
        directory: str = Query(...),
        forceLanguage: Union[str, None] = Query(default=None)
):
    transcribe_existing(directory, LanguageCode.from_string(forceLanguage))

# ============================================================================
# REFACTORED /ASR ENDPOINT WITH HASH-BASED DEDUPLICATION AND BLOCKING
# ============================================================================

@app.post("/asr")
async def asr(
    task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None),
    video_file: Union[str, None] = Query(default=None),
    initial_prompt: Union[str, None] = Query(default=None),
    audio_file: UploadFile = File(...),
    encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
    output: Union[str, None] = Query(default="srt", enum=["txt", "vtt", "srt", "tsv", "json"]),
    word_timestamps: bool = Query(default=False, description="Word-level timestamps"),
):
    """
    ASR endpoint that uses audio content hash for deduplication. 
    BLOCKS until processing is complete, then returns the result.
    
    If identical audio + task + language is already being processed,
    waits for that task to complete and returns the same result.
    """
    task_id = None
    
    try:
        logging.info(
            f"ASR {task.capitalize()} received for file '{video_file}'" 
            if video_file 
            else f"ASR {task.capitalize()} received"
        )
        
        # Read audio file content into memory
        file_content = await audio_file.read()
        
        if not file_content:
            await audio_file.close()
            return {
                "status": "error",
                "message": "Audio file is empty"
            }
        
        # Generate deterministic hash from audio (and optionally task/language)
        audio_hash = generate_audio_hash(file_content, task, language)
        task_id = f"asr-{audio_hash}"
        
        logging.debug(f"Generated audio hash: {audio_hash} for ASR request")
        
        # Handle forced language
        final_language = language
        if force_detected_language_to: 
            final_language = force_detected_language_to.to_iso_639_1()
            logging.info(f"Forcing detected language to {force_detected_language_to}")
        
        # Create result container for this task
        with task_results_lock:
            if task_id not in task_results:
                task_results[task_id] = TaskResult()
            task_result = task_results[task_id]
        
        # Queue the ASR task
        asr_task_data = {
            'path': task_id, # DeduplicatedQueue uses this for dedup
            'type': 'asr',
            'task': task,
            'language': final_language,
            'video_file': video_file,
            'initial_prompt': initial_prompt,
            'audio_content': file_content,
            'encode': encode,
            'output': output,
            'word_timestamps': word_timestamps,
            'result_container': task_result,
        }
        
        # Try to queue (returns False if already queued/processing)
        if task_queue.put(asr_task_data):
            logging.info(f"ASR task {task_id} queued")
        else:
            logging.info(f"ASR task {task_id} already queued/processing - waiting for result")
        
        # BLOCK HERE until worker completes (respects concurrent_transcriptions)
        if task_result.wait(timeout=asr_timeout):
            if task_result.error:
                logging.error(f"ASR task {task_id} failed: {task_result.error}")
                return {
                    "status": "error",
                    "task_id": task_id,
                    "message": f"ASR processing failed: {task_result.error}"
                }
            else: 
                logging.info(f"ASR task {task_id} completed")
                return StreamingResponse(
                    iter(task_result.result),
                    media_type="text/plain",
                    headers={'Source': f'{task.capitalize()}d using stable-ts from Subgen!'}
                )
        else:
            logging.error(f"ASR task {task_id} timed out")
            return {
                "status": "timeout",
                "task_id": task_id,
                "message": f"ASR processing timed out after {asr_timeout} seconds"
            }
            
    except Exception as e: 
        logging.error(f"Error in ASR endpoint: {e}", exc_info=True)
        return {"status": "error", "message": f"Error: {str(e)}"}
    finally:
        await audio_file.close()
        # MEMLEAK-FIX-1: Clean up task_results entry after task completes
        with task_results_lock:
            if task_id in task_results:
                del task_results[task_id]
                logging.debug(f"Cleaned up task_results entry for {task_id}")

# ============================================================================
# ASR WORKER FUNCTION
# ============================================================================

def asr_task_worker(task_data: dict) -> None:
    """
    Worker function that processes ASR tasks from the queue. 
    Called by transcription_worker when task type is 'asr'.
    """
    result = None
    task_id = task_data.get('path', 'unknown')
    result_container = task_data.get('result_container')
    
    try:
        task = task_data['task']
        language = task_data['language']
        video_file = task_data.get('video_file')
        initial_prompt = task_data.get('initial_prompt')
        file_content = task_data['audio_content']
        encode = task_data['encode']
        
        start_model()

        args = {}
        display_name = os.path.basename(video_file) if video_file else task_id
        args['progress_callback'] = ProgressHandler(display_name)
        
        # Handle audio encoding
        if encode:
            args['audio'] = file_content
        else:
            args['audio'] = np.frombuffer(file_content, np.int16).flatten().astype(np.float32) / 32768.0
            args['input_sr'] = 16000

        if custom_regroup and custom_regroup.lower() != 'default':
            args['regroup'] = custom_regroup

        args.update(kwargs)
        
        # Perform transcription
        result = model.transcribe(task=task, language=language, **args, verbose=None)
        appendLine(result)
        
        # Set result for blocking endpoint
        if result_container:
            result_container.set_result(result.to_srt_vtt(filepath=None, word_level=word_level_highlight))

    except Exception as e:
        logging.error(f"Error processing ASR (ID: {task_id}): {e}", exc_info=True)
        if result_container: 
            result_container.set_error(str(e))
    
    finally:
        delete_model()

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

# ============================================================================
# REFACTORED /DETECT-LANGUAGE ENDPOINT WITH HASH-BASED DEDUPLICATION AND BLOCKING
# ============================================================================

@app.post("/detect-language")
async def detect_language(
    audio_file: UploadFile = File(...),
    encode: bool = Query(default=True),
    video_file: Union[str, None] = Query(default=None),
    detect_lang_length: int = Query(default=detect_language_length),
    detect_lang_offset: int = Query(default=detect_language_offset)
):
    if force_detected_language_to: 
        await audio_file.close()
        return {"detected_language": force_detected_language_to.to_name(), "language_code": force_detected_language_to.to_iso_639_1()}
    
    try:
        file_content = await audio_file.read()
        if not file_content:
            return {"detected_language": "Unknown", "language_code": "und", "status": "error"}
            
        logging.info(f"Immediate language detection (Queue Bypass)" + (f" for {video_file}" if video_file else ""))
        
        # --- RUN IMMEDIATELY ---
        start_model()
        
        if encode:
            audio_bytes = extract_audio_segment_from_content(await audio_file.read(), detect_lang_offset, detect_lang_length)
            audio_data = np.frombuffer(audio_bytes, np.int16).flatten().astype(np.float32) / 32768.0
        else:
            audio_data = await get_audio_chunk(audio_file, detect_lang_offset, detect_lang_length)

        result = model.transcribe(audio_data, input_sr=16000, verbose=None)
        detected = LanguageCode.from_name(result.language)
        
        logging.info(f"Detect Language Result: {detected.to_name()} ({detected.to_iso_639_1()})")
        
        return {
            "detected_language": detected.to_name(),
            "language_code": detected.to_iso_639_1()
        }

    except Exception as e: 
        logging.error(f"Error in API detect-language: {e}", exc_info=True)
        return {"detected_language": "Unknown", "language_code": "und", "status": "error"}
    finally: 
        await audio_file.close()
        delete_model() # Schedules VRAM cleanup if system is idle

# ============================================================================
# DETECT LANGUAGE WORKER FOR UPLOADED AUDIO
# ============================================================================

def detect_language_from_upload(task_data: dict) -> None:
    """
    Worker function that processes detect-language tasks from uploaded audio. 
    Sets the result in the result_container when complete.
    """
    detected_language = LanguageCode.NONE
    task_id = task_data.get('path', 'unknown')
    result_container = task_data.get('result_container')
    
    try:
        video_file = task_data.get('video_file')
        file_content = task_data['audio_content']
        encode = task_data['encode']
        detect_lang_length = task_data['detect_lang_length']
        detect_lang_offset = task_data['detect_lang_offset']
        
        logging.info(
            f"Detecting language for '{video_file}' ({detect_lang_length}s starting at {detect_lang_offset}s) - ID: {task_id}"
            if video_file
            else f"Detecting language ({detect_lang_length}s starting at {detect_lang_offset}s) - ID: {task_id}"
        )
        
        start_model()

        args = {}
        args['progress_callback'] = progress
        
        # Handle audio extraction
        if encode:
            audio_bytes = extract_audio_segment_from_content(
                file_content, 
                detect_lang_offset, 
                detect_lang_length
            )
            args['audio'] = audio_bytes
            args['input_sr'] = 16000
        else:
            args['audio'] = np.frombuffer(file_content, np.int16).flatten().astype(np.float32) / 32768.0
            args['input_sr'] = 16000

        args.update(kwargs)
        
        detected_language = LanguageCode.from_name(model.transcribe(**args).language)
        language_code = detected_language.to_iso_639_1()
        
        logging.info(f"Detected language: {detected_language.to_name()} ({language_code}) - ID: {task_id}")
        
        # Set the result for the blocking endpoint
        if result_container:
            result_container.set_result({
                "detected_language": detected_language.to_name(),
                "language_code": language_code
            })

    except Exception as e:
        logging.error(
            f"Error detecting language (ID: {task_id}) for '{task_data.get('video_file')}': {e}"
            if task_data.get('video_file')
            else f"Error detecting language (ID: {task_id}): {e}",
            exc_info=True
        )
        if result_container: 
            result_container.set_error(str(e))
    
    finally:
        delete_model()

# ============================================================================
# HELPER: Extract audio segment from in-memory content
# ============================================================================

def extract_audio_segment_from_content(audio_content: bytes, start_time: int, duration: int) -> bytes:
    """
    Extract a segment of audio from in-memory content using FFmpeg.
    
    Args:
        audio_content: Raw audio bytes
        start_time: Start time in seconds
        duration: Duration in seconds
        
    Returns:
        Audio bytes of the extracted segment
    """
    try: 
        logging.info(f"Extracting audio segment: start_time={start_time}s, duration={duration}s")
        
        out, _ = (
            ffmpeg
            .input('pipe:0', ss=start_time, t=duration)
            .output('pipe:1', format='wav', acodec='pcm_s16le', ar=16000)
            .run(input=audio_content, capture_stdout=True, capture_stderr=True)
        )
        
        if not out:
            raise ValueError("FFmpeg output is empty")
        
        return out

    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error: {e.stderr.decode()}")
        return audio_content # Fallback to original if extraction fails
    except Exception as e:
        logging.error(f"Error extracting audio segment: {str(e)}")
        return audio_content # Fallback to original

def detect_language_task(path, original_task_data=None):
    """
    Worker function that detects language for a local file.
    Then queues the actual transcription with the detected language.
    """
    detected_language = LanguageCode.NONE
    
    try:
        logging.info(
            f"Detecting language of file: {path} "
            f"({detect_language_length}s starting at {detect_language_offset}s)"
        )
        
        start_model()
        
        audio_segment = extract_audio_segment_to_memory(
        # MEMLEAK-FIX-3: extract_audio_segment_to_memory now returns bytes directly
            path, 
            detect_language_offset, 
            int(detect_language_length)
        )
        
        detected_language = LanguageCode.from_name(model.transcribe(audio_segment).language)
        
        logging.info(f"Detected language: {detected_language.to_name()}")

    except Exception as e:
        logging.error(f"Error detecting language for file: {e}", exc_info=True)
        
    finally:
        delete_model()
        
        # Queue transcription with detected language
        task_data = {
            'path': path,
            'type': 'transcribe',
            'transcribe_or_translate': transcribe_or_translate,
            'force_language': detected_language
        }
        
        # Carry over metadata (Plex IDs, etc.) from the original task
        if original_task_data:
            for key, value in original_task_data.items():
                if key not in task_data:
                    task_data[key] = value
        
        if task_queue.put(task_data):
            logging.debug(f"Queued transcription for detected language: {path}")
        else:
            logging.debug(f"Transcription already queued/processing for: {path}")

def extract_audio_segment_to_memory(input_file, start_time, duration):
    """
    Extract a segment of audio from input_file, starting at start_time for duration seconds.
    
    :param input_file: UploadFile object or path to the input audio file
    :param start_time: Start time in seconds (e.g., 60 for 1 minute)
    :param duration: Duration in seconds (e.g., 30 for 30 seconds)
    :return: bytes containing the audio segment, or None on error
    
    MEMLEAK-FIX-3: Changed to return bytes directly instead of BytesIO to prevent memory leak.
    Previously returned BytesIO objects were never closed, causing 480KB-10MB leak per call.
    """
    try:
        if hasattr(input_file, 'file') and hasattr(input_file.file, 'read'): # Handling UploadFile
            input_file.file.seek(0) # Ensure the file pointer is at the beginning
            input_stream = 'pipe:0'
            input_kwargs = {'input': input_file.file.read()}
        elif isinstance(input_file, str): # Handling local file path
            input_stream = input_file
            input_kwargs = {}
        else:
            raise ValueError("Invalid input: input_file must be a file path or an UploadFile object.")

        logging.info(f"Extracting audio from: {input_stream}, start_time: {start_time}, duration: {duration}")

        # Run FFmpeg to extract the desired segment
        out, _ = (
            ffmpeg
            .input(input_stream, ss=start_time, t=duration) # Set start time and duration
            .output('pipe:1', format='wav', acodec='pcm_s16le', ar=16000) # Output to pipe as WAV
            .run(capture_stdout=True, capture_stderr=True, **input_kwargs)
        )

        # Check if the output is empty or null
        if not out:
            raise ValueError("FFmpeg output is empty, possibly due to invalid input.")
        
        # MEMLEAK-FIX-3: Return bytes directly instead of BytesIO to prevent memory leak
        return out

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

def schedule_model_cleanup():
    """Schedule model cleanup with a delay to allow concurrent requests."""
    global model_cleanup_timer, model_cleanup_lock
    
    with model_cleanup_lock:
        # Cancel any existing timer
        if model_cleanup_timer is not None:
            model_cleanup_timer.cancel()
            logging.debug("Cancelled previous model cleanup timer")
        
        # Schedule a new cleanup timer
        model_cleanup_timer = Timer(model_cleanup_delay, perform_model_cleanup)
        model_cleanup_timer.daemon = True
        model_cleanup_timer.start()
        logging.debug(f"Model cleanup scheduled in {model_cleanup_delay} seconds")

def perform_model_cleanup():
    """Actually perform the model cleanup."""
    global model, model_cleanup_timer, model_cleanup_lock
    
    with model_cleanup_lock: 
        logging.debug("Executing scheduled model cleanup")
        
        if clear_vram_on_complete and task_queue.is_idle():
            logging.debug("Queue idle; clearing model from memory.")
            if model: 
                try:
                    model.model.unload_model()
                    del model
                    model = None
                    logging.info("Model unloaded from memory")
                except Exception as e:
                    logging.error(f"Error unloading model: {e}")
            
            if transcribe_device.lower() == 'cuda' and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    logging.debug("CUDA cache cleared.")
                except Exception as e: 
                    logging.error(f"Error clearing CUDA cache: {e}")
        else:
            logging.debug("Queue not idle or clear_vram disabled; skipping model cleanup")
        
        if os.name != 'nt': # don't garbage collect on Windows
            gc.collect()
            ctypes.CDLL(ctypes.util.find_library('c')).malloc_trim(0)
        
        model_cleanup_timer = None

def delete_model():
    """
    Only schedules a cleanup timer if the system is actually idle.
    This prevents unnecessary timer resets when a large batch is being processed.
    """
    # 1. If we aren't supposed to clear VRAM, don't bother with timers at all.
    if not clear_vram_on_complete:
        return

    # 2. Only schedule cleanup if the queue is empty AND no other workers are processing.
    if task_queue.is_idle():
        schedule_model_cleanup()
    else:
        # If there are 10 items left in the queue, we simply do nothing. 
        # The very last worker to finish the last item will trigger the timer.
        logging.debug("Tasks still in queue or processing; skipping model cleanup scheduling.")

def isAudioFileExtension(file_extension):
    return file_extension.casefold() in AUDIO_EXTENSIONS

def write_lrc(result, file_path):
    with open(file_path, "w") as file:
        for segment in result.segments:
            minutes, seconds = divmod(int(segment.start), 60)
            fraction = int((segment.start - int(segment.start)) * 100)
            # remove embedded newlines in text, since some players ignore text after newlines
            text = segment.text[:].replace('\n', '')
            file.write(f"[{minutes:02d}:{seconds:02d}.{fraction:02d}]{text}\n")

def gen_subtitles(file_path: str, transcription_type: str, force_language: LanguageCode = LanguageCode.NONE) -> None:
    """Generates subtitles for a video file. 

    Args:
        file_path: str - The path to the video file. 
        transcription_type: str - The type of transcription or translation to perform.
        force_language: str - The language to force for transcription or translation. Default is None. 
    """

    try:
        start_model()
        
        # Check if the file is an audio file before trying to extract audio 
        file_name, file_extension = os.path.splitext(file_path)
        is_audio_file = isAudioFileExtension(file_extension)
        
        data = file_path
        # Extract audio from the file if it has multiple audio tracks
        extracted_audio_file = handle_multiple_audio_tracks(file_path, force_language)
        # MEMLEAK-FIX-3: handle_multiple_audio_tracks now returns bytes directly
        if extracted_audio_file:
            data = extracted_audio_file
        
        args = {}
        display_name = os.path.basename(file_path)
        args['progress_callback'] = ProgressHandler(display_name)
            
        if custom_regroup and custom_regroup.lower() != 'default':
            args['regroup'] = custom_regroup
            
        args.update(kwargs)
        
        result = model.transcribe(data, language=force_language.to_iso_639_1(), task=transcription_type, verbose=None, **args)

        appendLine(result)

        # If it is an audio file, write the LRC file
        if is_audio_file and lrc_for_audio_files:
            write_lrc(result, file_name + '.lrc')
        else:
            if not force_language: 
                force_language = LanguageCode.from_string(result.language)
            result.to_srt_vtt(name_subtitle(file_path, force_language), word_level=word_level_highlight)

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
        "NATIVE": lambda: language.to_name(in_english=False)
    }
    if transcribe_or_translate == 'translate':
        language = LanguageCode.ENGLISH
    return switch_dict.get(type, language.to_name)()

def name_subtitle(file_path: str, language: LanguageCode) -> str:
    """
    Name the subtitle file to be written, based on the source file and the language of the subtitle. 
    
    Args: 
        file_path: The path to the source file.
        language: The language of the subtitle.
    
    Returns:
        The name of the subtitle file to be written.
    """
    subgen_part = ".subgen" if show_in_subname_subgen else ""
    model_part = f".{whisper_model}" if show_in_subname_model else ""
    lang_part = define_subtitle_language_naming(language, subtitle_language_naming_type)
    
    return f"{os.path.splitext(file_path)[0]}{subgen_part}{model_part}.{lang_part}.srt"
    
def handle_multiple_audio_tracks(file_path: str, language: LanguageCode | None = None) -> bytes | None:
    """
    Handles the possibility of a media file having multiple audio tracks. 
    
    
    MEMLEAK-FIX-3: Changed to return bytes directly instead of BytesIO to prevent memory leak.
    Previously returned BytesIO objects were never closed, causing memory leaks.
    If the media file has multiple audio tracks, it will extract the audio track of the selected language. Otherwise, it will extract the first audio track.
    
    Parameters:
    file_path (str): The path to the media file.
    language (LanguageCode | None): The language of the audio track to search for. If None, it will extract the first audio track.
    
    Returns:
    bytes | None: The audio data as bytes, or None if no audio track was extracted.
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

def extract_audio_track_to_memory(input_video_path, track_index) -> bytes | None:
    """
    Extract a specific audio track from a video file to memory using FFmpeg. 

    Args:
        input_video_path (str): The path to the video file.
        track_index (int): The index of the audio track to extract. If None, skip extraction.

    Returns:
        bytes | None: The audio data as bytes, or None if extraction failed.
        
        MEMLEAK-FIX-3: Changed to return bytes directly instead of BytesIO to prevent memory leak.
        Previously returned BytesIO objects were never closed, causing memory leaks.
    """
    if track_index is None:
        logging.warning(f"Skipping audio track extraction for {input_video_path} because track index is None")
        return None

    try:
        # Use FFmpeg to extract the specific audio track and output to memory
        out, _ = (
            ffmpeg.input(input_video_path)
            .output(
                "pipe:", # Direct output to a pipe
                map=f"0:{track_index}", # Select the specific audio track
                format="wav", # Output format
                ac=1, # Mono audio (optional)
                ar=16000, # Sample rate 16 kHz (recommended for speech models)
                loglevel="quiet"
            )
            .run(capture_stdout=True, capture_stderr=True) # Capture output in memory
        )
        # MEMLEAK-FIX-3: Return bytes directly instead of BytesIO to prevent memory leak
        return out

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
    
    #logger.debug(f"choose_transcribe_language({file_path}, {forced_language})")
    
    if forced_language: 
        logger.debug(f"ENV FORCE_LANGUAGE is set: Forcing language to {forced_language}") 
        return forced_language

    if force_detected_language_to: 
        logger.debug(f"ENV FORCE_DETECTED_LANGUAGE_TO is set: Forcing detected language to {force_detected_language_to}")
        return force_detected_language_to

    audio_tracks = get_audio_tracks(file_path)
    
    preferred_track_language = find_language_audio_track(audio_tracks, preferred_audio_languages)

    if preferred_track_language: 
        #logging.debug(f"Preferred language found: {preferred_track_language}")
        return preferred_track_language
    
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
        find_languages (list): A list language codes to search for.
    
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
    
def gen_subtitles_queue(file_path: str, transcription_type: str, force_language: LanguageCode = LanguageCode.NONE, **kwargs) -> None:
    global task_queue
    
    # Check if this file is already in the queue or being processed
    if task_queue.is_active(file_path):
        logging.debug(f"Ignored: {os.path.basename(file_path)} is already queued or processing.")
        return
    
    if not has_audio(file_path):
        logging.debug(f"{file_path} doesn't have any audio to transcribe!")
        return
    
    force_language = choose_transcribe_language(file_path, force_language)

    if should_skip_file(file_path, force_language): # skip a file before we waste time detecting it's language
        return
    
    # check if we would like to detect audio language in case of no audio language specified. Will return here again with specified language from whisper
    if not force_language and should_whiser_detect_audio_language:
        # make a detect language task
        task_id = {'path': file_path, 'type': "detect_language"}
        # Pass metadata info (kwargs) to the detect task
        task_id.update(kwargs)
        task_queue.put(task_id)
        #logging.debug(f"Added to queue: {task_id['path']} [type: {task_id.get('type', 'transcribe')}]")
        return

    task = {
        'path': file_path,
        'transcribe_or_translate': transcription_type,
        'force_language': force_language
    }
    # Pass metadata info (kwargs) to the transcribe task
    task.update(kwargs)
    
    task_queue.put(task)
    #logging.debug(f"Added to queue: {task['path']}, {task['transcribe_or_translate']}, {task['force_language']}")

def should_skip_file(file_path: str, target_language: LanguageCode) -> bool:
    """
    Determines if subtitle generation should be skipped for a file.

    Args:
        file_path: Path to the media file.
        target_language: The desired language for transcription.

    Returns:
        True if the file should be skipped, False otherwise.
    """
    base_name = os.path.basename(file_path)
    file_name, file_ext = os.path.splitext(base_name)
    if transcribe_or_translate == 'translate': 
        target_language = LanguageCode.ENGLISH # Force our target language as english if we are translating
    # 1. Skip if it's an audio file and an LRC file already exists.
    if isAudioFileExtension(file_ext) and lrc_for_audio_files:
        lrc_path = os.path.join(os.path.dirname(file_path), f"{file_name}.lrc")
        if os.path.exists(lrc_path):
            logging.info(f"Skipping {base_name}: LRC file already exists.")
            return True

    # 2. Skip if language detection failed and we are configured to skip unknowns.
    if skip_unknown_language and target_language == LanguageCode.NONE:
        logging.info(f"Skipping {base_name}: Unknown language and skip_unknown_language is enabled.")
        return True

    # 3. Skip if a subtitle already exists in the target language.
    if skip_if_to_transcribe_sub_already_exist and has_subtitle_language(file_path, target_language):
        lang_name = target_language.to_name()
        logging.info(f"Skipping {base_name}: Subtitles already exist in {lang_name}.")
        return True

    # 4. Skip if an internal subtitle exists in skipifinternalsublang language.
    if skipifinternalsublang and has_subtitle_language_in_file(file_path, skipifinternalsublang):
        lang_name = skipifinternalsublang.to_name()
        logging.info(f"Skipping {base_name}: Internal subtitles in {lang_name} already exist.")
        return True

    # 5. Skip if an external subtitle exists in the namesublang language
    if skipifexternalsub and namesublang and LanguageCode.is_valid_language(namesublang):
        external_lang = LanguageCode.from_string(namesublang)
        if has_subtitle_of_language_in_folder(file_path, external_lang):
            lang_name = external_lang.to_name()
            logging.info(f"Skipping {base_name}: External subtitles in {lang_name} already exist.")
            return True

    # 6. Skip if any subtitle language is in the skip list.
    if any(lang in skip_lang_codes_list for lang in get_subtitle_languages(file_path)):
        logging.info(f"Skipping {base_name}: Contains a skipped subtitle language.")
        return True

    # 7. Audio track checks
    audio_langs = get_audio_languages(file_path)

    # 7a. Limit to preferred audio languages
    if limit_to_preferred_audio_languages:
        if not any(lang in preferred_audio_languages for lang in audio_langs):
            preferred_names = [lang.to_name() for lang in preferred_audio_languages]
            logging.info(f"Skipping {base_name}: No preferred audio tracks found (looking for {', '.join(preferred_names)})")
            return True

    # 7b. Skip if the audio track language is in the skip list
    if any(lang in skip_if_audio_track_is_in_list for lang in audio_langs):
        logging.info(f"Skipping {base_name}: Contains a skipped audio language.")
        return True

    #logging.debug(f"Processing {base_name}: No skip conditions met.")
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
            if target_language is LanguageCode.NONE:
                if skip_if_language_is_not_set_but_subtitles_exist and subtitle_streams:
                    logging.debug("Language is not set, but internal subtitles exist.")
                    return True
                if only_skip_if_subgen_subtitle:
                    #logging.debug("Skipping since only external subgen subtitles are considered.")
                    return False  # Skip if only looking for external subgen subtitles

            # Check if any subtitle stream matches the target language
            for stream in subtitle_streams:
                # Convert the subtitle stream's language to a LanguageCode instance and compare
                stream_language = LanguageCode.from_string(stream.metadata.get('language', '').lower())
                if stream_language == target_language:
                    #logging.debug(f"Subtitles in '{target_language}' language found in the video.")
                    return True

            #logging.debug(f"No subtitles in '{target_language}' language found in the video.")
            return False

    except Exception as e:
        logging.error(f"An error occurred while checking the file with pyav: {type(e).__name__}: {e}")
        return False

def has_subtitle_of_language_in_folder(video_file: str, target_language: LanguageCode, recursion: bool = True, only_skip_if_subgen_subtitle: bool = False) -> bool:
    """Checks if the given folder has a subtitle file with the given language.
    Args:
        video_file (str): The path of the video file.
        target_language (LanguageCode): The language of the subtitle file to search for.
        recursion (bool): If True, search subfolders. If False, only the current folder.
        only_skip_if_subgen_subtitle (bool): If True, only skip if subtitles are auto-generated ("subgen").
    Returns:
        bool: True if a matching subtitle file is found, False otherwise.
    """
    subtitle_extensions = {'.srt', '.vtt', '.sub', '.ass', '.ssa', '.idx', '.sbv', '.pgs', '.ttml', '.lrc'}

    video_folder = os.path.dirname(video_file)
    video_name = os.path.splitext(os.path.basename(video_file))[0]

    # logging.debug(f"Searching for subtitles in: {video_folder}")

    for file_name in os.listdir(video_folder):
        file_path = os.path.join(video_folder, file_name)

        # If it's a file and has a subtitle extension
        if os.path.isfile(file_path) and file_path.endswith(tuple(subtitle_extensions)):
            subtitle_name, ext = os.path.splitext(file_name)

            # Ensure the subtitle name starts with the video name
            if not subtitle_name.startswith(video_name):
                continue

            # Extract parts after video filename
            subtitle_parts = subtitle_name[len(video_name):].lstrip(".").split(".")

            # Check for "subgen"
            has_subgen = "subgen" in subtitle_parts

            # Special handling if only skipping for subgen subtitles
            if target_language == LanguageCode.NONE:
                if only_skip_if_subgen_subtitle:
                    if has_subgen:
                        logging.debug("Skipping subtitles because they are auto-generated ('subgen').")
                        return False
                logging.debug("Skipping subtitles because language is NONE.")
                return True  # Default behavior if subtitles exist

            # Check if the subtitle file matches the target language
            if is_valid_subtitle_language(subtitle_parts, target_language):
                if only_skip_if_subgen_subtitle and not has_subgen:
                    continue  # Ignore non-subgen subtitles if flag is set
                logging.debug(f"Found matching subtitle: {file_name} for language {target_language.name} (subgen={has_subgen})")
                return True

        # Recursively search subfolders
        elif os.path.isdir(file_path) and recursion:
            if has_subtitle_of_language_in_folder(os.path.join(file_path, os.path.basename(video_file)), target_language, False, only_skip_if_subgen_subtitle):
                return True

    return False

def is_valid_subtitle_language(subtitle_parts: List[str], target_language: LanguageCode) -> bool:
    """Checks if any part of the subtitle name matches the target language."""
    return any(LanguageCode.from_string(part) == target_language for part in subtitle_parts)

def get_next_plex_episode(current_episode_rating_key, stay_in_season: bool = False):
    """
    Get the next episode's ratingKey based on the current episode in Plex.
    Args:
        current_episode_rating_key (str): The ratingKey of the current episode.
        stay_in_season (bool): If True, only find the next episode within the current season.
                              If False, find the next episode in the series.
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
        episodes_in_season = len(episodes) #episodes.get('size') # changed from episodes.get("size") because size is not available

        # Find the current episode index and get the next one
        current_episode_number = None
        current_season_number = None
        next_season_number = None
        for episode in episodes:
            if episode.get("ratingKey") == current_episode_rating_key:
                current_episode_number = int(episode.get("index"))
                current_season_number = episode.get("parentIndex")
                break
            #if rating_key_element is None:
            #    logging.warning(f"ratingKey not found for episode at index")
            #    continue

        # Logic to find the next episode
        if stay_in_season:
          if current_episode_number == episodes_in_season:
              return None # End of season
          for episode in episodes:
            if int(episode.get("index")) == int(current_episode_number)+1:
                return episode.get("ratingKey")
        else: # Not staying in season, find the next overall episode
          # Find next season if it exists
          for season in seasons:
              if int(season.get("index")) == int(current_season_number)+1:
                  #print(f"next season is: {episode.get('ratingKey')}")
                  #print(season.get("title"))
                  next_season_number = season.get("ratingKey")
                  break

          if current_episode_number == episodes_in_season: # changed to episodes_in_season from int(episodes_in_season)
              if next_season_number is not None:
                logging.debug("At end of season, try to find next season and first episode.")
                url = f"{plexserver}/library/metadata/{next_season_number}/children"
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                episodes = ET.fromstring(response.content).findall(".//Video")
                current_episode_number = 0
              else:
                return None
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
    url = f"{server_ip}/Items/{itemid}/Refresh?MetadataRefreshMode=FullRefresh"

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
    logging.info(f"Threads: {str(whisper_threads)}, Concurrent transcriptions: {str(concurrent_transcriptions)}")
    logging.info(f"Transcribe device: {transcribe_device}, Model: {whisper_model}")
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    if transcribe_folders:
        transcribe_existing(transcribe_folders)
    uvicorn.run("__main__:app", host="0.0.0.0", port=int(webhookport), reload=reload_script_on_change, use_colors=True)
