#!/bin/bash

set -e

if [ "$#" -eq 1 ]; then
    BASE_DIR="$1"
else
    echo "Default path: /home/tm/Documents/universita/tirocinio_hipert/research/datasets/squeaky_17-04-26"
    echo "No starting directory provided. Using default path? (y/n)"
    read -r answer
    if [[ "$answer" == "y" ]]; then
        BASE_DIR="/home/tm/Documents/universita/tirocinio_hipert/research/datasets/squeaky_17-04-26"
    else
        echo "Usage: $0 <base_directory>"
        exit 1
    fi
fi

source ../.venv_utils/bin/activate

python3 build_from_bag.py "$BASE_DIR"

echo "Dataset built successfully from bag."

deactivate