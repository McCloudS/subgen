import os
import sys
import urllib.request
import subprocess
import argparse

def convert_to_bool(in_bool):
    # Convert the input to string and lower case, then check against true values
    return str(in_bool).lower() in ('true', 'on', '1', 'y', 'yes')

def install_packages_from_requirements(requirements_file):
    try:
        # Try installing with pip3
        subprocess.run(['pip3', 'install', '-r', requirements_file, '--upgrade'], check=True)
        print("Packages installed successfully using pip3.")
    except subprocess.CalledProcessError:
        try:
            # If pip3 fails, try installing with pip
            subprocess.run(['pip', 'install', '-r', requirements_file, '--upgrade'], check=True)
            print("Packages installed successfully using pip.")
        except subprocess.CalledProcessError:
            print("Failed to install packages using both pip3 and pip.")

def download_from_github(url, output_file):
    try:
        with urllib.request.urlopen(url) as response, open(output_file, 'wb') as out_file:
            data = response.read()  # a `bytes` object
            out_file.write(data)
        print(f"File downloaded successfully to {output_file}")
    except urllib.error.HTTPError as e:
        print(f"Failed to download file from {url}. HTTP Error Code: {e.code}")
    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason}")
    except Exception as e:
        print(f"An error occurred: {e}")

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
    # Check if the script is run with 'python' or 'python3'
    if 'python3' in sys.executable:
        python_cmd = 'python3'
    elif 'python' in sys.executable:
        python_cmd = 'python'
    else:
        print("Script started with an unknown command")
        sys.exit(1)
    if sys.version_info[0] < 3:
        print(f"This script requires Python 3 or higher, you are running {sys.version}")
        sys.exit(1)  # Terminate the script
    
    #Make sure we're saving subgen.py and subgen.env in the right folder
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', default=False, action='store_true', help="Enable console debugging (default: False)")
    parser.add_argument('-i', '--install', default=False, action='store_true', help="Install/update all necessary packages (default: False)")
    parser.add_argument('-a', '--append', default=False, action='store_true', help="Append 'Transcribed by whisper' to generated subtitle (default: False)")
    parser.add_argument('-u', '--update', default=False, action='store_true', help="Update Subgen (default: False)")
    parser.add_argument('-x', '--exitearly', default=False, action='store_true', help="Exit without running subgen.py (default: False)")
    parser.add_argument('-s', '--setupbazarr', default=False, action='store_true', help="Prompt for common Bazarr setup parameters and save them for future runs (default: False)")
    parser.add_argument('-b', '--branch', type=str, default='main', help='Specify the branch to download from. (default: main)') 

    args = parser.parse_args()

    # Set environment variables based on the parsed arguments
    os.environ['DEBUG'] = str(args.debug)
    os.environ['APPEND'] = str(args.append)

    if args.setupbazarr: 
        prompt_and_save_bazarr_env_variables()
    load_env_variables()

    # URL to the requirements.txt file on GitHub
    requirements_url = "https://raw.githubusercontent.com/McCloudS/subgen/main/requirements.txt"
    requirements_file = "requirements.txt"

    # Install packages from requirements.txt if the install or packageupdate argument is True
    if args.install:
        download_from_github(requirements_url, requirements_file)
        install_packages_from_requirements(requirements_file)
    
    # Get the branch name from the BRANCH environment variable or default to 'main'
    branch_name = args.branch or os.getenv("BRANCH", "main")

    # Determine the script name based on the branch name
    subgen_script_name = f"subgen-{branch_name}.py" if branch_name != "main" else "subgen.py"

    # Check if the script exists or if the UPDATE environment variable is set to True
    if not os.path.exists(subgen_script_name) or convert_to_bool(os.getenv("UPDATE", "False")):
        print(f"Downloading {subgen_script_name} from GitHub branch {branch_name}...")
        download_from_github(f"https://raw.githubusercontent.com/McCloudS/subgen/{branch_name}/subgen.py", subgen_script_name)
    else:
        print("File exists and UPDATE is not set to True, skipping download.")
        
    if not args.exitearly:
        if branch_name:
            subprocess.run([f'{python_cmd}', '-u', f'subgen-{branch_name}.py'], check=True)
        else:
            subprocess.run([f'{python_cmd}', '-u', 'subgen.py'], check=True)
    else:
        print("Not running subgen.py: -x or --exitearly set")

if __name__ == "__main__":
    main()
