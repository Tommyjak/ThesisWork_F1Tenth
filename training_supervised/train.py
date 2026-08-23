import sys
from ultralytics import YOLO
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import preprocessing.configs.exp_yaml_parser as eyp
from preprocessing.pipeline import run_preprocessing

BASE_DIR = Path(__file__).resolve().parent
REF_DIR = BASE_DIR / "references"
MODEL_PATH = BASE_DIR / "models/yolo26n.pt"

def train_model(config):
    print("Model loading...")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")

    print("Starting training...")
    results = model.train(
        data = config,

        # --- core ---
        epochs = 100,
        patience = 20,          # early stopping: stop if no improvement for 20 epochs
        imgsz = 640,            # standard, good balance speed/accuracy
        batch = -1,             # ultralytics auto-selects based on GPU VRAM

        # --- optimizer ---
        optimizer = 'AdamW',    # more stable than SGD for fine-tuning
        lr0 = 0.001,            # lower than default (0.01) since we're fine-tuning
        lrf = 0.01,             # final lr = lr0 * lrf -> 0.00001
        warmup_epochs = 3,      # gradual lr warmup at the start
        weight_decay = 0.0005,

        # --- loss weights ---
        cls = 0.3,             # less emphasis on classification (only 1 class)

        # --- augmentation ---
        hsv_h = 0.015,          # hue shift (keep low, indoor lighting is consistent)
        hsv_s = 0.5,            # saturation
        hsv_v = 0.5,            # brightness/value (more useful: handles shadow variation)
        fliplr = 0.5,           # horizontal flip (opponent can come from either side)
        flipud = 0.0,           # vertical flip (disable: cars don't appear upside down)
        mosaic = 1.0,           # very useful: simulates multiple opponents/distances
        mixup = 0.1,            # light mixup
        scale = 0.5,            # scale jitter (simulates distance variation)
        translate = 0.1,
        degrees = 5.0,          # small rotation (camera is roughly fixed on car)
        perspective = 0.0001,   # subtle perspective warp

        # --- evaluation ---
        iou = 0.5,              # IoU threshold for mAP computation (COCO standard)
        conf = 0.25,            # confidence threshold during validation

        # --- hardware ---
        device = 'cpu',
        workers = 8,

        # --- reproducibility & logging ---
        seed = 42,
        pretrained = True,
        exist_ok = True,
        project = './results/f1tenth_experiments',
    )
    print("Training completed.")
    return results

def run_training_experiments():
    for exp in REF_DIR.iterdir():
        if not exp.is_dir():
            continue

        config = exp / "config.yaml"

        train_model(config)

def main():
    run_preprocessing(REF_DIR)
    run_training_experiments()

if __name__ == "__main__":
    main()