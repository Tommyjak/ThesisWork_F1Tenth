import yaml
from pathlib import Path
import ds_yaml_parser as dyp
import exp_yaml_parser as eyp
import shutil


DS_REF_DIR = Path("../ds_references")
MRGD_DS_REF_DIR = Path("../merged_datasets")

def get_ds_refs_paths(names):
    train_configs, val_configs, test_configs = [], [], []

    for name in names:
        ds_refs_path = DS_REF_DIR / name
        print(f"[INFO] Checking dataset reference path for {name} at {ds_refs_path}...")

        if not ds_refs_path.exists():
            print(f"[WARNING] Dataset reference path for {name} does not exist at {ds_refs_path}. Please create it.")
        else:
            print(f"[INFO] Dataset reference path for {name} exists at {ds_refs_path}.")

            config_path = ds_refs_path / "config.yaml"
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                train_configs.append(config['train'])
                val_configs.append(config['val'])
                test_configs.append(config['test'])

    return train_configs, val_configs, test_configs  
    
def clean_merged_dir():
    if MRGD_DS_REF_DIR.exists():
        for item in MRGD_DS_REF_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item)

def create_merged_dataset_folder(desired_name):
    merged_dir = MRGD_DS_REF_DIR / f"{desired_name}_merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    return merged_dir

def create_yaml_config_for_merged_ds(merged_dir, train_paths, val_paths, test_paths):
    config = {
        'train': [str(Path(p).resolve()) for p in train_paths],
        'val': [str(Path(p).resolve()) for p in val_paths],
        'test': [str(Path(p).resolve()) for p in test_paths],
        'names': {0: 'racecar'}
    }

    config_path = merged_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

def create_merged_dataset():
    datasets_by_exp = eyp.get_exp_datasets()
    clean_merged_dir()
    for exp_idx, ds_names in enumerate(datasets_by_exp):
        train_configs, val_configs, test_configs = get_ds_refs_paths(ds_names)
        desired_name = f"exp{exp_idx+1}"
        merged_dir = create_merged_dataset_folder(desired_name)
        create_yaml_config_for_merged_ds(merged_dir, train_configs, val_configs, test_configs)

if __name__ == "__main__":
    create_merged_dataset()