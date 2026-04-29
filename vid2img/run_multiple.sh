#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

DIR="$1"

if [ ! -d "$DIR" ]; then
    echo "Error: '$DIR' is not a valid directory"
    exit 1
fi

MP4_FILES=("$DIR"/*.mp4)

if [ ! -e "${MP4_FILES[0]}" ]; then
    echo "No .mp4 files found in '$DIR'"
    exit 1
fi

echo "Found ${#MP4_FILES[@]} .mp4 file(s) in '$DIR'"
echo "---"

for video in "${MP4_FILES[@]}"; do
    echo "Processing: $video"
    python3 vid2img.py "$video"
    echo "Done: $video"
    echo "---"
done

echo "All videos processed."