#!/bin/bash

# Install the program
pip3 install numpy stable-ts flask faster-whisper requests
mkdir /subgen
cd /subgen
curl --no-clobber https://raw.githubusercontent.com/McCloudS/subgen/main/subgen/subgen.py --output /subgen/subgen.py
rm -rf /subgen/subgen.py.*
chmod 777 /subgen/subgen.py

# Start the program

python3 -u subgen.py
