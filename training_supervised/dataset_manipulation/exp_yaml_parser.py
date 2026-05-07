import yaml
from pathlib import Path

EXP_YAML_PATH = Path("./experiments.yaml")

def load_experiments_yaml(experiments_yaml_path=EXP_YAML_PATH):
    experiments = []
    with open(experiments_yaml_path, 'r') as f:
        experiments_yaml = yaml.safe_load(f)["experiments"]
        experiments_objs = list(experiments_yaml.items())
    
    for _, exp in experiments_objs:
        experiments.append(exp)

    return experiments

def get_exp_infos(experiments_yaml_path=EXP_YAML_PATH):
    experiments = load_experiments_yaml(experiments_yaml_path)
    return [
        {
            'name': exp['name'],
            'description': exp['description'],
            'datasets': exp['datasets'],
            'pcts': (exp['train_pct'], exp['val_pct'], exp['test_pct']),
            'seed': exp['seed'],
        }
        for exp in experiments
    ]

def get_exp_names():
    experiments = load_experiments_yaml(EXP_YAML_PATH)
    return [exp['name'] for exp in experiments]

def get_exp_descriptions():
    experiments = load_experiments_yaml(EXP_YAML_PATH)
    return [exp['description'] for exp in experiments]

def get_exp_datasets():
    experiments = load_experiments_yaml(EXP_YAML_PATH)
    return [exp['datasets'] for exp in experiments]

def get_exp_pcts():
    experiments = load_experiments_yaml(EXP_YAML_PATH)
    return [exp['train_pct'] for exp in experiments], [exp['val_pct'] for exp in experiments], [exp['test_pct'] for exp in experiments]

def get_exp_seeds():
    experiments = load_experiments_yaml(EXP_YAML_PATH)
    return [exp['seed'] for exp in experiments]
