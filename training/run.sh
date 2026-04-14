#!/bin/bash

# TODO: fix this script to run in a cleaner way
set -e

source ../.venv_utils/bin/activate

python3 build_ds.py complete
python3 train.py run_complete

python3 build_ds.py low_bg
python3 train.py run_low_bg

python3 build_ds.py no_bg
python3 train.py run_no_bg

deactivate