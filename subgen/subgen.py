import sys
import os
import time
import json
import glob
import pathlib
import webhook_listener
import subprocess

# parse our arguments from the Dockerfile
whisper_model = sys.argv[1]
whisper_speedup = sys.argv[2]
whisper_threads = sys.argv[3]
whisper_processors = sys.argv[4]
procaddedmedia = sys.argv[5]
procmediaonplay = sys.argv[6]
namesublang = sys.argv[7]
updaterepo = sys.argv[8]

def process_post_request(request, *args, **kwargs):
    print("Received a webhook!")
    if int(request.headers.get('Content-Length', 0)) > 0:
        body = request.body.read(
            int(request.headers['Content-Length'])).decode()
    else:
        body = '{}'

    print(body)
    fullpath = json.loads(body)['file']
    filename = json.loads(body)['filename']
    event = json.loads(body)['event']
    filepath = os.path.dirname(fullpath)
    extension = pathlib.Path(filename).suffix
    filenamenoextension = filename.replace(extension, "")

    print("fullpath: " + fullpath)
    print("filename: " + filename)
    print("filepath: " + filepath)
    print("extension: " + extension)
    print("file name with no extension: " + filenamenoextension)
    print("event: " + event)

    if ((procaddedmedia and event == "added") or (procmediaonplay and event == "played")) and (len(glob.glob("{}/{}*subgen*".format(filepath, filenamenoextension))) == 0) and not os.path.isfile("{}.output.wav".format(fullpath)): #glob nonsense checks if there exists a subgen file already and won't make a new one
        if os.getenv('WHISPER_SPEEDUP') == "True" :
            print("This is a speedup run!")
            finalsubname = "{0}/{1}.subgen.{2}.speedup.{3}".format(
                filepath, filenamenoextension, whisper_model, namesublang)
        else:
            print("No speedup")
            finalsubname = "{0}/{1}.subgen.{2}.{3}".format(
                filepath, filenamenoextension, whisper_model, namesublang)
                
        gen_subtitles(fullpath, "{}.output.wav".format(fullpath), finalsubname)

        if os.path.isfile("{}.output.wav".format(fullpath)):
            print("Deleting WAV workfile")
            os.remove("{}.output.wav".format(fullpath))

    return

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

if not os.path.isdir("/whisper.cpp"):
    os.mkdir("/whisper.cpp")
os.chdir("/whisper.cpp")
subprocess.call("git clone https://github.com/ggerganov/whisper.cpp .", shell=True)
if os.getenv('UPDATEREPO') == "True":
    print("Updating repo!")
    subprocess.call("git pull", shell=True)
if os.path.isfile("/whisper.cpp/samples/jfk.wav"):
    print("Deleting sample file")
    os.remove("/whisper.cpp/samples/jfk.wav")
subprocess.call("make " + whisper_model, shell=True)
print("Starting webhook!")
webhooks = webhook_listener.Listener(handlers={"POST": process_post_request})
webhooks.start()
print("Webhook started")

while True:
    print("Still alive...")
    time.sleep(300)
