#!/bin/bash

# Install the program
pip3 install numpy stable-ts flask faster-whisper requests
wget -P /subgen https://raw.githubusercontent.com/McCloudS/subgen/main/subgen/subgen.py

# Start the program
cd /subgen
python -u subgen.py