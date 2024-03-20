import os
import requests
import subprocess
import argparse

def convert_to_bool(in_bool):
    # Convert the input to string and lower case, then check against true values
    return str(in_bool).lower() in ('true', 'on', '1')

def install_packages_from_requirements(requirements_file, force_update):
    try:
        # If force_update is True, add the '--upgrade' flag to force an update
        if force_update:
            subprocess.run(['pip3', 'install', '-r', requirements_file, '--upgrade'], check=True)
        else:
            subprocess.run(['pip3', 'install', '-r', requirements_file], check=True)
        
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

def main():
    # Construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False, action='store_true', help="Enable console debugging (default: False)")
    parser.add_argument('--install', default=False, action='store_true', help="Install packages (default: False)")
    parser.add_argument('--packageupdate', default=False, action='store_true', help="Force update PIP packages (default: False)")
    parser.add_argument('--append', default=False, action='store_true', help="Append 'Transcribed by whisper' to generated subtitle (default: False)")
    parser.add_argument('--update', default=False, action='store_true', help="Update Subgen (default: False)")
                  
    args = parser.parse_args()

    # Set environment variables based on the parsed arguments
    os.environ['DEBUG'] = str(args.debug)
    os.environ['APPEND'] = str(args.append)

    # URL to the requirements.txt file on GitHub
    requirements_url = "https://raw.githubusercontent.com/McCloudS/subgen/main/requirements.txt"
    requirements_file = "requirements.txt"

    # Install packages from requirements.txt if the install or packageupdate argument is True
    if args.install or args.packageupdate:
        install_packages_from_requirements(requirements_file, args.packageupdate)
    
    subgen_script_name = "./subgen.py"
    
    if not os.path.exists(subgen_script_name):
        print(f"File {subgen_script_name} does not exist. Downloading from GitHub...")
        download_from_github("https://raw.githubusercontent.com/McCloudS/subgen/main/subgen/subgen.py", subgen_script_name)
    elif convert_to_bool(os.getenv("UPDATE", "False")) or args.update:
        print(f"File exists, but UPDATE is set to True. Downloading {subgen_script_name} from GitHub...")
        download_from_github("https://raw.githubusercontent.com/McCloudS/subgen/main/subgen/subgen.py", subgen_script_name)
    else:
        print("Environment variable UPDATE is not set or set to False, skipping download.")
        
    subprocess.run(['python3', '-u', 'subgen.py'], check=True)

if __name__ == "__main__":
    main()
