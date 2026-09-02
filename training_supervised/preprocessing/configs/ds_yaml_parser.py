import yaml
from pathlib import Path

DS_YAML_PATH = Path(__file__).resolve().parent / "datasets.yaml"

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
            'root': Path(ds['root']),
            'type': ds['type'],
        }
        for ds in datasets
    ]

def get_ds_names():
    datasets = load_datasets_yaml(DS_YAML_PATH)
    return [ds['name'] for ds in datasets]

def get_ds_roots():
    datasets = load_datasets_yaml(DS_YAML_PATH)
    return [ds['root'] for ds in datasets]

def get_ds_types():
    datasets = load_datasets_yaml(DS_YAML_PATH)
    return [ds['type'] for ds in datasets]
