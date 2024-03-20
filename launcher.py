import os
import requests
import subprocess
import argparse

def convert_to_bool(in_bool):
    # Convert the input to string and lower case, then check against true values
    return str(in_bool).lower() in ('true', 'on', '1')

def install_packages_from_requirements(requirements_file):
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

def main():
    # Construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument( '-d', '--debug', default=False, action='store_true', help="Enable console debugging (default: False)")
    parser.add_argument('-i', '--install', default=False, action='store_true', help="Install/update all necessary packages (default: False)")
    parser.add_argument('-a', '--append', default=False, action='store_true', help="Append 'Transcribed by whisper' to generated subtitle (default: False)")
    parser.add_argument('-u', '--update', default=False, action='store_true', help="Update Subgen (default: False)")
    parser.add_argument('-s', '--skiprun', default=False, action='store_true', help="Skip running subgen.py (default: False)")
                  
    args = parser.parse_args()

    # Set environment variables based on the parsed arguments
    os.environ['DEBUG'] = str(args.debug)
    os.environ['APPEND'] = str(args.append)

    # URL to the requirements.txt file on GitHub
    requirements_url = "https://raw.githubusercontent.com/McCloudS/subgen/main/requirements.txt"
    requirements_file = "requirements.txt"

    # Install packages from requirements.txt if the install or packageupdate argument is True
    if args.install:
        install_packages_from_requirements(requirements_file)
    
    subgen_script_name = "./subgen.py"
    
    if not os.path.exists(subgen_script_name):
        print(f"File {subgen_script_name} does not exist. Downloading from GitHub...")
        download_from_github("https://raw.githubusercontent.com/McCloudS/subgen/main/subgen.py", subgen_script_name)
    elif convert_to_bool(os.getenv("UPDATE", "False")) or args.update:
        print(f"File exists, but UPDATE is set to True. Downloading {subgen_script_name} from GitHub...")
        download_from_github("https://raw.githubusercontent.com/McCloudS/subgen/main/subgen.py", subgen_script_name)
    else:
        print("Environment variable UPDATE is not set or set to False, skipping download.")
    if not args.skiprun:    
        subprocess.run(['python3', '-u', 'subgen.py'], check=True)
    else:
        print("not running subgen.py: -s or --skiprun set")

if __name__ == "__main__":
    main()
