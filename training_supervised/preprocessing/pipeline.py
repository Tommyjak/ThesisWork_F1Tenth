from preprocessing.loading import load_datasets
import preprocessing.configs.exp_yaml_parser as eyp
from preprocessing.composition import build_experiments
from preprocessing.export import write_experiments

def run_preprocessing(out_dir):
    datasets = load_datasets()

    train_multi, val_multi = [], []
    
    train_multi, val_multi = build_experiments(datasets)

    write_experiments(train_multi, val_multi, out_dir)
    