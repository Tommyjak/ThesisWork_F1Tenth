import sys
from ultralytics import YOLO
import yaml
import  merge_datasets as md
import ds_yaml_parser as dyp
import exp_yaml_parser as eyp
import create_refs_from_ds as crfd
from pathlib import Path

DS_REF_DIR = Path("../ds_references")
MRGD_DS_REF_DIR = Path("../merged_datasets")
MODEL_PATH = Path("../models/yolo26n.pt")

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
        device = 0,
        workers = 8,

        # --- reproducibility & logging ---
        seed = 42,
        pretrained = True,
        exist_ok = True,
        project = './results/f1tenth_experiments',
    )
    print("Training completed.")
    return results

def get_configs_for_training():
    configs = []
    for subfolder in sorted(MRGD_DS_REF_DIR.iterdir()):
        if not subfolder.is_dir():
            continue
    
        yaml_file = subfolder / "config.yaml"
    
        if not yaml_file.exists():
            print(f"[WARNING] No config.yaml found in {subfolder.name}, skipping.")
            continue
    
        configs.append(yaml_file)

    return configs
    
def run_multiple_experiments():
    crfd.create_refs_from_ds()
    md.create_merged_dataset()

    configs = get_configs_for_training()
    summary = []

    for config in sorted(configs):
        print(f"Training with config: {config}...")
        results = train_model(Path(config))
        summary.append({
            'experiment': config.parent.name,
            'mAP50': results.results_dict['metrics/mAP50(B)'],
            'save_dir': str(results.save_dir)
        })

    print("\n=== Experiment Summary ===")
    for s in summary:
        print(f"{s['experiment']}: mAP50={s['mAP50']:.4f} | saved at {s['save_dir']}")


def main():
    run_multiple_experiments()

if __name__ == "__main__":
    main()