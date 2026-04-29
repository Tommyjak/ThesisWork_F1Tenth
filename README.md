# Thesis Work for Degree in AIE - Unimore
## Unsupervised Learning for Obstacle Detection in Autonomous Driving (F1Tenth - RoboRacer) on Edge Devices
This repository contains the core of the thesis work for the degree in Artificial Intelligence and Engineering (AIE) at the University of Modena and Reggio Emilia (Unimore). The project focuses on unsupervised learning techniques for obstacle detection in autonomous driving, specifically using the F1Tenth - RoboRacer platform.

## Project Structure
The project, as of today, 29/04/2026, is organized as follows:
```text
.
├── README.md
├── requirements.txt
├── training_supervised
│   ├── analyze_results.py
│   ├── background_comparison_dashboard.png
│   ├── build_ds.py
│   ├── build_from_bag.py
│   ├── build_from_bag.sh
│   ├── config.yaml
│   ├── inference.py  
│   ├── run.sh
│   └── train.py
├── training_unsupervised/
└── vid2img/
    ├── run_multiple.sh
    ├── run.sh
    └── vid2img.py
    
```

## Description of Directories
- `vid2img/`: Contains scripts for converting video data into image frames, creating a dataset for training
  * `vid2img.py`: Script for converting video files into image frames
  * `run.sh`: Shell script to execute the video to image conversion process 
  * `run_multiple.sh`: Shell script to execute the video to image conversion process for multiple videos
- `training_supervised/`: Contains scripts and resources for supervised training of the obstacle detection
  * `build_ds.py`: Script for building a dataset from multiple ones, being able to select percentages of "negative" samples (i.e., with no detections of obstacles)
  * `build_from_bag.py`: Script for building a dataset from multiple directories containing files extracted from ROS bags
  * `build_from_bag.sh`: Shell script to execute the dataset building process from ROS
  * `train.py`: Main training script for supervised learning
  * `inference.py`: Script for running inference using the trained model
  * `analyze_results.py`: Script for analyzing the results of different runs of training and generating visual comparative analysis
  * `config.yaml`: Configuration file for training parameters and datasets paths
  * `run.sh`: Shell script to execute the training process
- `training_unsupervised/`: Placeholder for future unsupervised training scripts and resources