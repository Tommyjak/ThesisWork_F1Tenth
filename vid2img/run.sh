#!/bin/bash

case "$1" in
    --help)
        echo "USAGE: $0 <path_to_video_file>"
        exit 0
        ;;
    *)
        if [ -f "$1" ]; then
            echo "Starting Python script to extract frames from video: $1"
        else
            echo "Error: File '$1' does not exist."
            echo "Usage: $0 <path_to_video_file>"
            exit 1
        fi
        ;;
esac

# Activate the virtual environment
source ../.venv_utils/bin/activate

# Run the Python script
python3 vid2img.py "$1"

deactivate