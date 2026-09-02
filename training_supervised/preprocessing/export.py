from pathlib import Path
import yaml
import shutil

import preprocessing.configs.exp_yaml_parser as eyp

def write_experiments(train_multi, val_multi, out_dir):
    out_dir = Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (train, val) in enumerate(zip(train_multi, val_multi)):
        exp_name = eyp.get_exp_names()[i]
        exp_dir = out_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        train_file = exp_dir / "train.txt"
        val_file = exp_dir / "val.txt"
        config_file = exp_dir / "config.yaml"

        files = (train_file, val_file)

        for file in files:
            file.touch(exist_ok=True)
            with open(file, 'w') as f:
                if file == train_file:
                    for sample in train:
                        f.write(str(sample.img_path) + '\n')
                elif file == val_file:
                    for sample in val:
                        f.write(str(sample.img_path) + '\n')

        config = {
            'train': str(train_file),
            'val': str(val_file),
            'names': {0: 'racecar'}
        }

        with open(config_file, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
