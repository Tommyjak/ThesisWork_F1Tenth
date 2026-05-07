import yaml
from pathlib import Path

DS_YAML_PATH = Path("./datasets.yaml")

def load_datasets_yaml(datasets_yaml_path):
    datasets = []
    with open(datasets_yaml_path, 'r') as f:
        datasets_yaml = yaml.safe_load(f)["datasets"]
        datasets_objs = list(datasets_yaml.items())
    
    for _, ds in datasets_objs:
        datasets.append(ds)

    return datasets

def get_ds_infos(datasets_yaml_path=DS_YAML_PATH):
    datasets = load_datasets_yaml(datasets_yaml_path)
    return [
        {
            'name': ds['name'],
            'description': ds['description'],
            'mixed': ds['mixed'] == 1,
            'already_split': ds['already_split'] == 1,
            'base_path': Path(ds['base_path']),
        }
        for ds in datasets
    ]

def get_ds_names():
    datasets = load_datasets_yaml(DS_YAML_PATH)
    return [ds['name'] for ds in datasets]

def get_ds_descriptions():
    datasets = load_datasets_yaml(DS_YAML_PATH)
    return [ds['description'] for ds in datasets]

def get_ds_mixed_bool():
    datasets = load_datasets_yaml(DS_YAML_PATH)
    return [ds['mixed'] == 1 for ds in datasets]

def get_already_split_bool():
    datasets = load_datasets_yaml(DS_YAML_PATH)
    return [ds['already_split'] == 1 for ds in datasets]

def get_ds_base_paths():
    datasets = load_datasets_yaml(DS_YAML_PATH)
    return [ds['base_path'] for ds in datasets]

def extract_name_from_path(path):
    path = Path(path)
    parts = path.parts
    return parts[parts.index("datasets") + 1]