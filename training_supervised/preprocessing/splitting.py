from core.samples import Split

def split_pool(samples, val_pct, rng):
    pool = list(samples)
    rng.shuffle(pool)

    n_val = round(len(pool) * val_pct)

    train = pool[n_val :]
    val = pool[: n_val]

    for t in train:
        t.split = Split.TRAIN

    for v in val:
        v.split = Split.VAL

    return train, val