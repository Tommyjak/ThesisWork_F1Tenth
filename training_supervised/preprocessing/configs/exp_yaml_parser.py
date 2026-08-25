import yaml
from pathlib import Path

EXP_YAML_PATH = Path(__file__).resolve().parent / "experiments.yaml"

def load_experiments_yaml(experiments_yaml_path=EXP_YAML_PATH):
    with open(experiments_yaml_path, 'r') as f:
        experiments_yaml = yaml.safe_load(f)["experiments"]
    return list(experiments_yaml.values())

def get_exp_infos(experiments_yaml_path=EXP_YAML_PATH):
    experiments = load_experiments_yaml(experiments_yaml_path)
    return [
        {
            'name': exp['name'],
            'datasets': exp['datasets'],
            'ratios': {
                name: (cfg.get('pos_ratio', 1.0), cfg.get('neg_ratio', 1.0))
                for name, cfg in exp['datasets'].items()
            },
            'val_pct': exp['val_pct'],
            'seed': exp['seed'],
        }
        for exp in experiments
    ]

def get_exp_names():
    experiments = load_experiments_yaml(EXP_YAML_PATH)
    return [exp['name'] for exp in experiments]

def get_exp_datasets():
    experiments = load_experiments_yaml(EXP_YAML_PATH)
    return [exp['datasets'] for exp in experiments]

def get_ratios():
    experiments = load_experiments_yaml(EXP_YAML_PATH)
    return [
        [
            (cfg.get('pos_ratio', 1.0), cfg.get('neg_ratio', 1.0))
            for cfg in exp['datasets'].values()
        ]
        for exp in experiments
    ]

def get_val_pcts():
    experiments = load_experiments_yaml(EXP_YAML_PATH)
    return [exp['val_pct'] for exp in experiments]

def get_exp_seeds():
    experiments = load_experiments_yaml(EXP_YAML_PATH)
    return [exp['seed'] for exp in experiments]