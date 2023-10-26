#!/bin/bash

# Install the program
pip3 install numpy stable-ts flask faster-whisper requests
cd /subgen
curl /subgen https://raw.githubusercontent.com/McCloudS/subgen/main/subgen/subgen.py

# Start the program

python3 -u subgen.py
