import os
import requests

def convert_to_bool(in_bool):
    if isinstance(in_bool, bool):
        return in_bool
    else:
        value = str(in_bool).lower()
        return value not in ('false', 'off', '0', 0)

def download_from_github(url, output_file):
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded successfully to {output_file}")
    else:
        print(f"Failed to download file from {url}")

def main():
    github_url = "https://raw.githubusercontent.com/McCloudS/subgen/main/subgen/subgen.py"
    output_file = "./subgen.py"
    
    # Check if the environment variable is set
    github_download_enabled = convert_to_bool(os.getenv("UPDATE", False))
    
    if not os.path.exists(output_file):
        print(f"File {output_file} does not exist. Downloading from GitHub...")
        download_from_github(github_url, output_file)
    elif github_download_enabled:
        print(f"File exists, but UPDATE is set to True. Downloading {output_file} from GitHub...")
        download_from_github(github_url, output_file)
    else:
        print("Environment variable UPDATE is not set or set to False, skipping download.")

if __name__ == "__main__":
    main()
