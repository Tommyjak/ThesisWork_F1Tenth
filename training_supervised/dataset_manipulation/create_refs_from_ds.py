import random
from pathlib import Path
import yaml
from . import ds_yaml_parser as dyp
import shutil

# TODO: avoid hardcoding this path, maybe pass it as an argument or read from a config file
DS_REF_DIR = Path("../ds_references")

def create_dataset_structure(desired_name):
    """Create output directories for train, val, and test sets."""
    ref_dir = DS_REF_DIR / desired_name

    ref_dir.mkdir(parents=True, exist_ok=True)

    return ref_dir

def generate_path_splits(source_dir, train_pct = 0.7, val_pct = 0.15, test_pct = 0.15, seed: int = 42):
    """
    Randomly samples and splits items from a directory into train, val, and test sets.

    Args:
        source_dir: Path to the source directory
        train_pct: Fraction of items for training (default: 0.7)
        val_pct: Fraction of items for validation (default: 0.15)
        test_pct: Fraction of items for testing (default: 0.15)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Dictionary with keys 'train', 'val', 'test' mapping to lists of Paths
    """

    dir_name = source_dir / Path("images")
    items = list(Path(dir_name).iterdir())

    if not items:
        raise ValueError(f"No items found in {source_dir}")

    random.seed(seed)
    random.shuffle(items)

    n = len(items)
    train_end = int(n * train_pct)
    val_end = train_end + int(n * val_pct)

    return {
        "train": items[:train_end],
        "val":   items[train_end:val_end],
        "test":  items[val_end:],
    }

def get_splits_from_ds(source_path):
    splits = {}
    for split in ['train', 'val', 'test']:
        images_dir = source_path / Path(split) / Path("images")
        items = list(Path(images_dir).iterdir())
        splits[split] = items

    return splits 

def write_paths_to_txt(paths, txt_dst):
    with open(txt_dst, 'w') as f:
        for path in paths:
            f.write(f"{path.resolve()}\n")

def create_yolo_config(ref_dir):
    train_txt_path = ref_dir / "train.txt"
    val_txt_path = ref_dir / "val.txt"
    test_txt_path = ref_dir / "test.txt"

    yolo_config = {
        'train': str(train_txt_path),
        'val': str(val_txt_path),
        'test': str(test_txt_path),
        'names': {0: 'racecar'}
    }

    yaml_path = ref_dir / "config.yaml"
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yolo_config, f, sort_keys=False)

def is_already_split(source_path):
    expected_dirs = ['train', 'val', 'test']
    for d in expected_dirs:
        path = source_path / Path(d)
        if not path.exists():
            print(f"Directory {d} not found in {source_path}. Dataset is not already split.")
            return False
    return True

def clean_directory():
    if DS_REF_DIR.exists():
        for item in DS_REF_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item)

def create_refs(source_path, train_pct=0.7, val_pct=0.15, test_pct=0.15):
    if train_pct + val_pct + test_pct != 1:
        raise ValueError("Train, validation, and test percentages must sum to 1.")
    
    desired_name = dyp.extract_name_from_path(source_path)
    
    ref_dir = create_dataset_structure(desired_name)
    
    if is_already_split(source_path):
        print(f"Dataset at {source_path} is already split. Extracting paths...")
        splits = get_splits_from_ds(source_path)
        print(f"Extracted splits: { {k: len(v) for k, v in splits.items()} }")
    else:
        print(f"Dataset at {source_path} is not split. Creating new splits...")
        splits = generate_path_splits(source_path, train_pct, val_pct, test_pct)
        print(f"Splits created and saved in {DS_REF_DIR / desired_name}")

    for name in ['train', 'val', 'test']:
        txt_path = ref_dir / f"{name}.txt"
        write_paths_to_txt(splits[name], txt_path)

    create_yolo_config(ref_dir)

def create_refs_from_ds():
    clean_directory()
    ds_paths = dyp.get_ds_base_paths()
    for ds_path in ds_paths:
        print(f"Processing dataset: {ds_path}...")
        create_refs(ds_path, train_pct=0.7, val_pct=0.15, test_pct=0.15)
    