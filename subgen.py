subgen_version = '2024.3.24.60'

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
from fastapi.responses import StreamingResponse, RedirectResponse, HTMLResponse
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

def update_env_variables():
    global plextoken, plexserver, jellyfintoken, jellyfinserver, whisper_model, whisper_threads
    global concurrent_transcriptions, transcribe_device, procaddedmedia, procmediaonplay
    global namesublang, skipifinternalsublang, webhookport, word_level_highlight, debug
    global use_path_mapping, path_mapping_from, path_mapping_to, model_location, monitor
    global transcribe_folders, transcribe_or_translate, force_detected_language_to
    global clear_vram_on_complete, compute_type, append, reload_script_on_change
    global model_prompt, custom_model_prompt, lrc_for_audio_files, custom_regroup
    
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
    custom_regroup = os.getenv('CUSTOM_REGROUP', 'cm_sl=84_sl=42++++++1')

    if transcribe_device == "gpu":
        transcribe_device = "cuda"

    set_env_variables_from_file('subgen.env')

app = FastAPI()
model = None
files_to_transcribe = []
subextension =  f".subgen.{whisper_model.split('.')[0]}.{namesublang}.srt"
subextensionSDH =  f".subgen.{whisper_model.split('.')[0]}.{namesublang}.sdh.srt"

# Assuming 'env_variables' is a dictionary containing your environment variables
# and their default values, as well as descriptions.

# Function to read environment variables from a file and return them as a dictionary
def get_env_variables_from_file(filename):
    env_vars = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    env_vars[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"File {filename} not found. Using default values.")
    return env_vars
    
def set_env_variables_from_file(filename):
    try:
        with open(filename, 'r') as file:
            for line in file:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('\"').strip("'")
    except FileNotFoundError:
        print(f"File {filename} not found. Environment variables not set.")

# Function to generate HTML form with values filled from the environment file
@app.get("/", response_class=HTMLResponse)
def form_get():
    # Read the environment variables from the file
    env_values = get_env_variables_from_file('subgen.env')
    html_content = "<html><head><title>Subgen settings!</title></head><body>"
    html_content += '<img src="https://raw.githubusercontent.com/McCloudS/subgen/main/icon.png" alt="Header Image" style="display: block; margin-left: auto; margin-right: auto; width: 10%;">'
    html_content += "<html><body><form action=\"/submit\" method=\"post\">"
    
    for var_name, var_info in env_variables.items():
        # Use the value from the environment file if it exists, otherwise use the default
        value = env_values.get(var_name, str(var_info['default']))
        html_content += f"<br><div><strong>{var_name}</strong>: {var_info['description']} (<strong>default: {var_info['default']}</strong>)<br>"
        if var_name == "TRANSCRIBE_OR_TRANSLATE":
        # Add a dropdown for TRANSCRIBE_OR_TRANSLATE with options 'Transcribe' and 'Translate'
        	selected_value = value if value in ['transcribe', 'translate'] else var_info['default']
        	html_content += f"<select name=\"{var_name}\">"
        	html_content += f"<option value=\"transcribe\"{' selected' if selected_value == 'transcribe' else ''}>Transcribe</option>"
        	html_content += f"<option value=\"translate\"{' selected' if selected_value == 'translate' else ''}>Translate</option>"
        	html_content += "</select><br>"
        elif isinstance(var_info['default'], bool):
            # Convert the value to a boolean
            value_bool = value.lower() == 'true'
            # Determine the selected value based on the current value or the default value
            selected_value = value_bool if value in ['True', 'False'] else var_info['default']
            html_content += f"<select name=\"{var_name}\">"
            html_content += f"<option value=\"True\"{' selected' if selected_value else ''}>True</option>"
            html_content += f"<option value=\"False\"{' selected' if not selected_value else ''}>False</option>"
            html_content += "</select><br>"
        else:
            html_content += f"<input type=\"text\" name=\"{var_name}\" value=\"{env_values.get(var_name, '') if var_name in env_values else ''}\" placeholder=\"{var_info['default']}\"/></div>"

    html_content += "<br><input type=\"submit\" value=\"Save\"/></form></body></html>"
    return html_content

        return {"detected_language": whisper_languages.get(detected_lang_code, detected_lang_code) , "language_code": detected_lang_code}

@app.post("/submit")
async def form_post(request: Request):
    env_path = 'subgen2.env'
    form_data = await request.form()
    # Read the existing content of the file
    try:
        with open(f"{env_path}", "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        lines = []

    # Create a dictionary of existing variables
    existing_vars = {}
    for line in lines:
        if "=" in line:
            var, val = line.split("=", 1)
            existing_vars[var.strip()] = val.strip()

    # Update the file with new values from the form
    with open(f"{env_path}", "w") as file:
        for key, value in form_data.items():
            # Normalize the key to uppercase
            key = key.upper()
            # Convert the value to the correct type (boolean or string)
            if isinstance(env_variables[key]['default'], bool):
                value = value.strip().lower() == 'true'
            else:
                value = value.strip()
            # Write to file only if the value is different from the default
            if env_variables[key]["default"] != value and value:
                # Check if the variable already exists and if the value is different
                if key in existing_vars and existing_vars[key] != str(value):
                    # Update the existing variable with the new value
                    existing_vars[key] = str(value)
                elif key not in existing_vars:
                    # Add the new variable to the dictionary
                    existing_vars[key] = str(value)
            elif key in existing_vars:
                # Remove the entry from the existing variables if the value is empty
                del existing_vars[key]
                del os.environ[key]

        # Write the updated variables to the file
        for var, val in existing_vars.items():
            file.write(f"{var}={val}\n")
    update_env_variables()
    return(f"Configuration saved to {env_path}, reloading your subgen with your new values!")

    Returns:
        The full path to the file.
    """

    url = f"{server_ip}/library/metadata/{itemid}"

env_variables = {
    "TRANSCRIBE_DEVICE": {"description": "Can transcribe via gpu (Cuda only) or cpu. Takes option of 'cpu', 'gpu', 'cuda'.", "default": "cpu", "value": ""},
    "WHISPER_MODEL": {"description": "Can be: 'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1','large-v2', 'large-v3', 'large', 'distil-large-v2', 'distil-medium.en', 'distil-small.en'", "default": "medium", "value": ""},
    "CONCURRENT_TRANSCRIPTIONS": {"description": "Number of files it will transcribe in parallel", "default": "2", "value": ""},
    "WHISPER_THREADS": {"description": "Number of threads to use during computation", "default": "4", "value": ""},
    "MODEL_PATH": {"description": "This is where the WHISPER_MODEL will be stored. This defaults to placing it where you execute the script in the folder 'models'", "default": "./models", "value": ""},
    "PROCADDEDMEDIA": {"description": "Will gen subtitles for all media added regardless of existing external/embedded subtitles (based off of SKIPIFINTERNALSUBLANG)", "default": True, "value": ""},
    "PROCMEDIAONPLAY": {"description": "Will gen subtitles for all played media regardless of existing external/embedded subtitles (based off of SKIPIFINTERNALSUBLANG)", "default": True, "value": ""},
    "NAMESUBLANG": {"description": "Allows you to pick what it will name the subtitle. Instead of using EN, I'm using AA, so it doesn't mix with existing external EN subs, and AA will populate higher on the list in Plex.", "default": "aa", "value": ""},
    "SKIPIFINTERNALSUBLANG": {"description": "Will not generate a subtitle if the file has an internal sub matching the 3 letter code of this variable", "default": "eng", "value": ""},
    "WORD_LEVEL_HIGHLIGHT": {"description": "Highlights each word as it's spoken in the subtitle.", "default": False, "value": ""},
    "PLEXSERVER": {"description": "This needs to be set to your local plex server address/port", "default": "http://plex:32400", "value": ""},
    "PLEXTOKEN": {"description": "This needs to be set to your plex token", "default": "token here", "value": ""},
    "JELLYFINSERVER": {"description": "Set to your Jellyfin server address/port", "default": "http://jellyfin:8096", "value": ""},
    "JELLYFINTOKEN": {"description": "Generate a token inside the Jellyfin interface", "default": "token here", "value": ""},
    "WEBHOOKPORT": {"description": "Change this if you need a different port for your webhook", "default": "9000", "value": ""},
    "USE_PATH_MAPPING": {"description": "Similar to sonarr and radarr path mapping, this will attempt to replace paths on file systems that don't have identical paths. Currently only support for one path replacement.", "default": False, "value": ""},
    "PATH_MAPPING_FROM": {"description": "This is the path of my media relative to my Plex server", "default": "/tv", "value": ""},
    "PATH_MAPPING_TO": {"description": "This is the path of that same folder relative to my Mac Mini that will run the script", "default": "/Volumes/TV", "value": ""},
    "TRANSCRIBE_FOLDERS": {"description": "Takes a pipe '|' separated list and iterates through and adds those files to be queued for subtitle generation if they don't have internal subtitles", "default": "", "value": ""},
    "TRANSCRIBE_OR_TRANSLATE": {"description": "Takes either 'transcribe' or 'translate'. Transcribe will transcribe the audio in the same language as the input. Translate will transcribe and translate into English.", "default": "transcribe", "value": ""},
    "COMPUTE_TYPE": {"description": "Set compute-type using the following information: https://github.com/OpenNMT/CTranslate2/blob/master/docs/quantization.md", "default": "auto", "value": ""},
    "DEBUG": {"description": "Provides some debug data that can be helpful to troubleshoot path mapping and other issues. If set to true, any modifications to the script will auto-reload it (if it isn't actively transcoding). Useful to make small tweaks without re-downloading the whole file.", "default": True, "value": ""},
    "FORCE_DETECTED_LANGUAGE_TO": {"description": "This is to force the model to a language instead of the detected one, takes a 2 letter language code.", "default": "", "value": ""},
    "CLEAR_VRAM_ON_COMPLETE": {"description": "This will delete the model and do garbage collection when queue is empty. Good if you need to use the VRAM for something else.", "default": True, "value": ""},
    "UPDATE": {"description": "Will pull latest subgen.py from the repository if True. False will use the original subgen.py built into the Docker image. Standalone users can use this with launcher.py to get updates.","default": False,"value": ""},
    "APPEND": {"description": "Will add the following at the end of a subtitle: 'Transcribed by whisperAI with faster-whisper ({whisper_model}) on {datetime.now()}'","default": False,"value": ""},
    "MONITOR": {"description": "Will monitor TRANSCRIBE_FOLDERS for real-time changes to see if we need to generate subtitles","default": False,"value": ""},
    "USE_MODEL_PROMPT": {"description": "When set to True, will use the default prompt stored in greetings_translations 'Hello, welcome to my lecture.' to try and force the use of punctuation in transcriptions that don't.","default": False,"value": ""},
    "CUSTOM_MODEL_PROMPT": {"description": "If USE_MODEL_PROMPT is True, you can override the default prompt (See: [prompt engineering in whisper](https://medium.com/axinc-ai/prompt-engineering-in-whisper-6bb18003562d%29) for great examples).","default": "","value": ""},
    "LRC_FOR_AUDIO_FILES": {"description": "Will generate LRC (instead of SRT) files for filetypes: '.mp3', '.flac', '.wav', '.alac', '.ape', '.ogg', '.wma', '.m4a', '.m4b', '.aac', '.aiff'","default": True,"value": ""},
    "CUSTOM_REGROUP": {"description": "Attempts to regroup some of the segments to make a cleaner looking subtitle. See #68 for discussion. Set to blank if you want to use Stable-TS default regroups algorithm of cm_sp=,* /，_sg=.5_mg=.3+3_sp=.* /。/?/？","default": "cm_sl=84_sl=42++++++1","value": ""}
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
    uvicorn.run("__main__:app", host="0.0.0.0", port=int(webhookport), reload=reload_script_on_change, use_colors=True)
