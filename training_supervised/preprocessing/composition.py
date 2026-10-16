import random

from preprocessing.splitting import split_pool
import preprocessing.configs.exp_yaml_parser as eyp

SPLIT_SEED = 1234

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
    val_pcts = eyp.get_val_pcts()

    train_multi, val_multi = [], []

    for i, _ in enumerate(exp_names):
        rng = random.Random(exp_seeds[i])
        val_pct = val_pcts[i]
        ds_requested = exp_datasets[i]

        train, val = [], []

        # un dataset richiesto dallo yaml degli esperimenti ma assente da quello
        # dei dataset produceva un training set VUOTO in silenzio (il ciclo sotto
        # salta i dataset non richiesti, non i richiesti mancanti). E' successo
        # con i seq_* e datasets.yaml di default: meglio fermarsi qui che a meta'
        # di una run da 100 epoche su zero immagini
        loaded = {ds.name for ds in datasets}
        missing = sorted(set(ds_requested) - loaded)

        if missing:
            raise KeyError(
                f"[{exp_names[i]}] dataset richiesti ma non definiti nello yaml "
                f"dei dataset: {missing}. Serve DS_YAML=tmp_datasets.yaml?"
            )

        for ds in datasets:
            if ds.name not in ds_requested:
                continue

            cfg = ds_requested[ds.name]

            pools = (
                ("pos", ds.get_positive_samples(), cfg.get("pos_ratio", 1.0)),
                ("neg", ds.get_negative_samples(), cfg.get("neg_ratio", 1.0)),
            )

            for kind, pool, ratio in pools:
                sub_rng = random.Random(f"{exp_seeds[i]}:{ds.name}:{kind}:{ratio}")
                pool = limit_samples(pool, ratio, sub_rng)

                split_rng = random.Random(f"{SPLIT_SEED}:{ds.name}:{kind}:{val_pct}")
                tr, va = split_pool(pool, val_pct, split_rng)

                train += tr
                val += va

        rng.shuffle(train)
        rng.shuffle(val)
        
        train_multi.append(train)
        val_multi.append(val)

    return train_multi, val_multi
