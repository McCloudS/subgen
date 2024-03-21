import os
import requests
import subprocess
import argparse

def convert_to_bool(in_bool):
    # Convert the input to string and lower case, then check against true values
    return str(in_bool).lower() in ('true', 'on', '1', 'y', 'yes')

def install_packages_from_requirements(requirements_file, force_update):
    try:
        subprocess.run(['pip3', 'install', '-r', requirements_file, '--upgrade'], check=True)
        print(f"Requirements from {requirements_file} have been successfully installed.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install requirements: {e}")

def download_from_github(url, output_file):
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded successfully to {output_file}")
    else:
        print(f"Failed to download file from {url}")

def prompt_and_save_bazarr_env_variables():
    """
    Prompts the user for Bazarr related environment variables with descriptions and saves them to a file.
    If the user does not input anything, default values are used.
    """
    # Instructions for the user
    instructions = (
        "You will be prompted for several configuration values.\n"
        "If you wish to use the default value for any of them, simply press Enter without typing anything.\n"
        "The default values are shown in brackets [] next to the prompts.\n"
        "Items can be the value of true, on, 1, y, yes, false, off, 0, n, no, or an appropriate text response.\n"
    )
    print(instructions)
    env_vars = {
        'WHISPER_MODEL': ('Whisper Model', 'Enter the Whisper model you want to run: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large, distil-large-v2, distil-medium.en, distil-small.en', 'medium'),
        'WEBHOOKPORT': ('Webhook Port', 'Default listening port for subgen.py', '9000'),
        'TRANSCRIBE_DEVICE': ('Transcribe Device', 'Set as cpu or gpu', 'gpu'),
        'DEBUG': ('Debug', 'Enable debug logging', 'True'),
        'CLEAR_VRAM_ON_COMPLETE': ('Clear VRAM', 'Attempt to clear VRAM when complete (Windows users may need to set this to False)', 'False'),
        'APPEND': ('Append', 'Append \'Transcribed by whisper\' to generated subtitle', 'False'),
    }

    # Dictionary to hold the user's input
    user_input = {}

    # Prompt the user for each environment variable and write to .env file
    with open('subgen.env', 'w') as file:
        for var, (description, prompt, default) in env_vars.items():
            value = input(f"{prompt} [{default}]: ") or default
            file.write(f"{var}={value}\n")

    print("Environment variables have been saved to subgen.env")

def load_env_variables(env_filename='subgen.env'):
    """
    Loads environment variables from a specified .env file and sets them.
    """
    try:
        with open(env_filename, 'r') as file:
            for line in file:
                var, value = line.strip().split('=', 1)
                os.environ[var] = value

        print(f"Environment variables have been loaded from {env_filename}")

    except FileNotFoundError:
        print(f"{env_filename} file not found. Please run prompt_and_save_env_variables() first.")

def main():
    #Make sure we're saving subgen.py and subgen.env in the right folder
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument( '-d', '--debug', default=False, action='store_true', help="Enable console debugging (default: False)")
    parser.add_argument('-i', '--install', default=False, action='store_true', help="Install/update all necessary packages (default: False)")
    parser.add_argument('-a', '--append', default=False, action='store_true', help="Append 'Transcribed by whisper' to generated subtitle (default: False)")
    parser.add_argument('-u', '--update', default=False, action='store_true', help="Update Subgen (default: False)")
    parser.add_argument('-dnr', '--donotrun', default=False, action='store_true', help="Do not run subgen.py (default: False)")
    parser.add_argument('-b', '--bazarrsetup', default=False, action='store_true', help="Prompt for common Bazarr setup parameters and save them for future runs (default: False)")

    
                  
    args = parser.parse_args()

    # Set environment variables based on the parsed arguments
    os.environ['DEBUG'] = str(args.debug)
    os.environ['APPEND'] = str(args.append)

    if args.bazarrsetup: 
        prompt_and_save_bazarr_env_variables()
    load_env_variables()

    # URL to the requirements.txt file on GitHub
    requirements_url = "https://raw.githubusercontent.com/McCloudS/subgen/main/requirements.txt"
    requirements_file = "requirements.txt"

    # Install packages from requirements.txt if the install or packageupdate argument is True
    if args.install:
        install_packages_from_requirements(requirements_file, args.install)
    
    subgen_script_name = "./subgen.py"
    
    if not os.path.exists(subgen_script_name):
        print(f"File {subgen_script_name} does not exist. Downloading from GitHub...")
        download_from_github("https://raw.githubusercontent.com/McCloudS/subgen/main/subgen.py", subgen_script_name)
    elif convert_to_bool(os.getenv("UPDATE", "False")) or args.update:
        print(f"File exists, but UPDATE is set to True. Downloading {subgen_script_name} from GitHub...")
        download_from_github("https://raw.githubusercontent.com/McCloudS/subgen/main/subgen.py", subgen_script_name)
    else:
        print("Environment variable UPDATE is not set or set to False, skipping download.")
    if not args.donotrun:    
        subprocess.run(['python3', '-u', 'subgen.py'], check=True)
    else:
        print("not running subgen.py: -dnr or --donotrun")

if __name__ == "__main__":
    main()
