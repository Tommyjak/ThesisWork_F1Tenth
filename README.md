# Thesis Work for Degree in AIE - Unimore
## Unsupervised Learning for Obstacle Detection in Autonomous Driving (F1Tenth - RoboRacer) on Edge Devices
This repository contains the core of the thesis work for the degree in Artificial Intelligence and Engineering (AIE) at the University of Modena and Reggio Emilia (Unimore). The project focuses on unsupervised learning techniques for obstacle detection in autonomous driving, specifically using the F1Tenth - RoboRacer platform.

## Project Structure
The project, as of today, 07/05/2026, is organized as follows:
```text
.
.
├── dataset_manipulation
│   ├── create_refs_from_ds.py
│   ├── datasets.yaml
│   ├── ds_yaml_parser.py
│   ├── experiments.yaml
│   ├── exp_yaml_parser.py
│   ├── inference.py
│   ├── __init__.py
│   ├── merge_datasets.py
│   ├── train.py
│   └── unify_bags_data.py
├── ds_references
│   ├── dataset_1
│   ├── ...
│   └── dataset_n
├── merged_datasets
│   ├── exp1_merged
│   ├── ...
│   └── exp_n_merged
├── models
    ├── exp1_merged
    ├── ...
    └── exp_n_merged


    
```

## Description of Directories
- `vid2img/`: Contains scripts for converting video data into image frames, creating a dataset for training
  * `vid2img.py`: Script for converting video files into image frames
  * `run.sh`: Shell script to execute the video to image conversion process 
  * `run_multiple.sh`: Shell script to execute the video to image conversion process for multiple videos
- `training_supervised/`: Contains scripts and resources for supervised training of the obstacle detection
  * `create_refs_from_ds.py`: Generates .txt split files and `config.yaml` references for each dataset under `ds_references/`
  * `datasets.yaml`: Declares the available datasets with their paths and metadata
  * `ds_yaml_parser.py`: Utility functions to load and query dataset entries from `datasets.yaml`
  * `experiments.yaml`: Declares training experiments, each defined by a dataset combination and split parameters
  * `exp_yaml_parser.py`: Utility functions to load and query experiment entries from `experiments.yaml`
  * `inference.py`: Opens the webcam and runs live opponent detection using a trained YOLO model
  * `merge_datasets.py`: Combines dataset references into per-experiment merged `config.yaml` files under `merged_datasets/`
  * `train.py`: Orchestrates the full training pipeline across all defined experiments
  * `unify_bags_data.py`: Merges multiple ROS bag extraction folders into a single unified `complete_labels/` dataset directory
- `training_unsupervised/`: Placeholder for possible future unsupervised training scripts and resources

## Usage
When using the project you have to browse to `training_supervised/dataset_manipulation/` folder
Then:
  1. If you want to add a new dataset, just add a new dataset, with all the specifics as the others in the `datasets.yaml` file. Make sure that the datasets are in the same parent folder even outside the project, called `datasets/`
  2. If you want to add a new experiment (specifying which and how many datasets to merge) you have to add it to `experiments.py` file. It's necessary to add an experiment to run the training with success.
  3. If you want to run the training, just run the command from terminal `python3 train.py`
  4. If you want to run the inference, just run the command from terminal `python3 inference.py`