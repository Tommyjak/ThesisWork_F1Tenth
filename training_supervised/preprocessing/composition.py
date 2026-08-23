import random

from preprocessing.splitting import split_pool
import preprocessing.configs.exp_yaml_parser as eyp

def limit_samples(samples, fraction, rng):
    samples = list(samples)
    if fraction >= 1.0:
        return samples
    else:
        rng.shuffle(samples)
        limited_len = round(len(samples) * fraction)
        return samples[: limited_len]
        
def build_experiments(datasets):
    exp_names = eyp.get_exp_names()
    exp_datasets = eyp.get_exp_datasets()
    exp_seeds = eyp.get_exp_seeds()
    train_pcts, val_pcts, test_pcts = eyp.get_exp_pcts()

    train_multi, val_multi, test_multi = [], [], []

    for i, _ in enumerate(exp_names):
        rng = random.Random(exp_seeds[i])
        pcts = (train_pcts[i], val_pcts[i], test_pcts[i])
        ds_requested = exp_datasets[i]

        train, val, test = [], [], []

        for ds in datasets:
            if ds.name not in ds_requested:
                continue

            cfg = ds_requested[ds.name]
            for pool in (limit_samples(ds.get_positive_samples(), cfg.get('pos_ratio', 1.0), rng), limit_samples(ds.get_negative_samples(), cfg.get('neg_ratio', 1.0), rng)):
                tr, va, te = split_pool(pool, pcts, rng)

                train += tr
                val += va
                test += te

        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)
        
        train_multi.append(train)
        val_multi.append(val)
        test_multi.append(test)

    return train_multi, val_multi, test_multi
