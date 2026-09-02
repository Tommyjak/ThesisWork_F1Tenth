from pathlib import Path

import benchmarking.configs.bm_yaml_parser as bmyp
from core.dataset import Dataset
from core.loaders.standard_loader import StandardLoader
from core.loaders.subfolders_loader import SubfoldersLoader

def load_benchmarks():
    benchmarks = []

    bm_infos = bmyp.get_bms_infos()

    for bm in bm_infos:
        if not bm['enabled']:
            continue

        if bm['type'] == 'standard':
            loader = StandardLoader(Path(bm['root']))
            pos_samples, neg_samples = loader.find_samples()

            if not pos_samples and not neg_samples:
                raise RuntimeError(f"Benchmark '{bm['name']}': nessuna immagine in {bm['root']}")

            benchmarks.append(Dataset(
                name = bm['name'],
                root = bm['root'],
                pos_samples = pos_samples,
                neg_samples = neg_samples
            ))
        elif bm['type'] == 'subfolders':
            loader = SubfoldersLoader(Path(bm['root']))
            pos_samples, neg_samples = loader.find_samples()

            if not pos_samples and not neg_samples:
                raise RuntimeError(f"Benchmark '{bm['name']}': nessuna immagine in {bm['root']}")

            benchmarks.append(Dataset(
                name = bm['name'],
                root = bm['root'],
                pos_samples = pos_samples,
                neg_samples = neg_samples
            ))
        else:
            raise ValueError(f"Unknown dataset type: {bm['type']}")

    return benchmarks