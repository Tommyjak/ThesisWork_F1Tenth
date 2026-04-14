import os
import yaml
import random
from pathlib import Path
import sys

def exec_selection(experiments):
    """Ask the user to select which configuration to use for training among the ones defined in the config file."""
    print(
        f"Type which configuration you want to use for training among the following options:\n"
        f"-> {experiments[0]['name']}: pct_ds1: {experiments[0]['pct_ds1']}, pct_ds2_pos: {experiments[0]['pct_ds2_pos']}, pct_ds2_neg: {experiments[0]['pct_ds2_neg']}\n"
        f"-> {experiments[1]['name']}: pct_ds1: {experiments[1]['pct_ds1']}, pct_ds2_pos: {experiments[1]['pct_ds2_pos']}, pct_ds2_neg: {experiments[1]['pct_ds2_neg']}\n"
        f"-> {experiments[2]['name']}: pct_ds1: {experiments[2]['pct_ds1']}, pct_ds2_pos: {experiments[2]['pct_ds2_pos']}, pct_ds2_neg: {experiments[2]['pct_ds2_neg']}\n"
        f"-> custom (you choose which percentage of negative samples of dataset2 to use)"
    )
    
    sel_ok = False

    while not sel_ok:
        selection = input("Enter your choice: ")
        if selection.lower() == experiments[0]['name']:
            pct_ds1 = experiments[0]['pct_ds1']
            pct_ds2_pos = experiments[0]['pct_ds2_pos']
            pct_ds2_neg = experiments[0]['pct_ds2_neg']
            sel_ok = True
        elif selection.lower() == experiments[1]['name']:
            pct_ds1 = experiments[1]['pct_ds1']
            pct_ds2_pos = experiments[1]['pct_ds2_pos']
            pct_ds2_neg = experiments[1]['pct_ds2_neg']
            sel_ok = True
        elif selection.lower() == experiments[2]['name']:
            pct_ds1 = experiments[2]['pct_ds1']
            pct_ds2_pos = experiments[2]['pct_ds2_pos']
            pct_ds2_neg = experiments[2]['pct_ds2_neg']
            sel_ok = True
        elif selection.lower() == "custom":
            pct_ok = False
            while not pct_ok:
                pct_ds2_neg = float(input("Enter the percentage of negative samples from dataset2 to use (between 0 and 1): "))
                if (0 <= pct_ds2_neg <= 1):
                    pct_ok = True
                else:
                    print("Percentage must be between 0 and 1.")


            pct_ds1 = 1.0
            pct_ds2_pos = 1.0
            sel_ok = True
        else:
            print(
                f"Invalid selection '{selection}'. Please choose one of the following options:\n" 
                f"-> {experiments[0]['name']};\n"
                f"-> {experiments[1]['name']};\n"
                f"-> {experiments[2]['name']};\n"
                f"-> custom"
            )
    
    return pct_ds1, pct_ds2_pos, pct_ds2_neg

def is_negative_sample(label_path):
    """Check if a label file is empty or missing."""
    if not os.path.exists(label_path):
        return True
    return os.path.getsize(label_path) == 0

def get_pairs(images_dir):
    labels_dir = images_dir.replace("images", "labels")
    positives=[]
    negatives=[]
    valid_exts = {
        '.jpg',
        '.jpeg',
        '.png'
    }

    for img_name in os.listdir(images_dir):
        if Path(img_name).suffix.lower() not in valid_exts:
            continue

        img_path = os.path.join(images_dir, img_name)
        label_name = Path(img_name).stem + '.txt'
        label_path = os.path.join(labels_dir, label_name)

        if is_negative_sample(label_path):
            negatives.append(img_path)
        else:
            positives.append(img_path)

    return positives, negatives

def extract_dataset(config_data, pct_ds1, pct_ds2_pos, pct_ds2_neg):
    """Extract the specified percentage of samples from the two datasets according to the provided configuration."""

    # Dataset paths extraction from YAML config file
    ds1 = config_data["datasets"]["dataset1"]
    ds2 = config_data["datasets"]["dataset2"]

    # Divide ds1 and ds2 into positive and negative samples
    ds1_pos, ds1_neg = get_pairs(ds1["train_images"])
    ds2_pos, ds2_neg = get_pairs(ds2["train_images"])

    # Sample from train dataset of dataset1 and dataset2 based on given percentages
    sampled_ds1 = random.sample(ds1_pos + ds1_neg, int(len(ds1_pos + ds1_neg) * pct_ds1))
    sampled_ds2_pos = random.sample(ds2_pos, int(len(ds2_pos) * pct_ds2_pos))
    sampled_ds2_neg = random.sample(ds2_neg, int(len(ds2_neg) * pct_ds2_neg))

    # Combine everything
    train_images = sampled_ds1 + sampled_ds2_pos + sampled_ds2_neg

    # Create a directory to store temporary files (only if it doesn't exist)
    Path("tmp").mkdir(parents=True, exist_ok=True)

    # Create a .txt file and write on it the paths of sampled images
    train_txt_path = Path("tmp/tmp_train.txt").resolve()

    with open(train_txt_path, 'w') as f:
        for img in train_images:
            f.write(f"{Path(img).resolve()}\n")

    # Create a YAML file for YOLO training with the paths to the train, val and test datasets
    yaml_path = Path("tmp/tmp_config.yaml").resolve()

    yolo_config = {
        'train': str(train_txt_path),
        'val': str(Path(ds1["val_images"]).resolve()),
        'test': str(Path(ds1["test_images"]).resolve()),
        'names': {0: 'racecar'}
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(yolo_config, f, sort_keys=False)

if __name__ == "__main__":
    config_file = "config.yaml"

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file '{config_file}' not found.")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    #pct_ds1, pct_ds2_pos, pct_ds2_neg = exec_selection(config["experiments"])

    if len(sys.argv) > 1:
        selection = sys.argv[1].lower()

        selected_exp = None
        for exp in config["experiments"]:
            if exp["name"] == selection:
                selected_exp = exp
                break

        if selected_exp is not None:
            pct_ds1 = selected_exp["pct_ds1"]
            pct_ds2_pos = selected_exp["pct_ds2_pos"]
            pct_ds2_neg = selected_exp["pct_ds2_neg"]

        else:
            print(f"Error: Configuration '{selection}' not found in config.yaml.")
            sys.exit(1)
    else:
        pct_ds1, pct_ds2_pos, pct_ds2_neg = exec_selection(config["experiments"])

    extract_dataset(config, pct_ds1, pct_ds2_pos, pct_ds2_neg)