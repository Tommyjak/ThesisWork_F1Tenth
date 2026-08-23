import random
from preprocessing.core.samples import Split

def split_pool(samples, pcts, rng):
    pool = list(samples)
    rng.shuffle(pool)

    n_train = round(len(pool) * pcts[0])
    n_val = round(len(pool) * pcts[1])

    train = pool[: n_train]
    val = pool[n_train : n_train +n_val]
    test = pool[n_train + n_val :]

    for t in train:
        t.split = Split.TRAIN

    for v in val:
        v.split = Split.VAL

    for t in test:
        t.split = Split.TEST

    return train, val, test