from pathlib import Path

import preprocessing.configs.ds_yaml_parser as dyp
from preprocessing.core.dataset import Dataset
from preprocessing.loaders.standard_loader import StandardLoader
from preprocessing.loaders.subfolders_loader import SubfoldersLoader

def load_datasets():
    datasets = []

    ds_infos = dyp.get_ds_infos()

    for ds in ds_infos:
        if ds['type'] == 'standard':
            loader = StandardLoader(Path(ds['root']))
            pos_samples, neg_samples = loader.find_samples()
            datasets.append(Dataset(
                name = ds['name'],
                root = ds['root'],
                pos_samples = pos_samples,
                neg_samples = neg_samples
            ))
        elif ds['type'] == 'subfolders':
            loader = SubfoldersLoader(Path(ds['root']))
            pos_samples, neg_samples = loader.find_samples()
            datasets.append(Dataset(
                name = ds['name'],
                root = ds['root'],
                pos_samples = pos_samples,
                neg_samples = neg_samples
            ))
        else:
            raise ValueError(f"Unknown dataset type: {ds['type']}")

    return datasets