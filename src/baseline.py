"""Independent-column baseline."""
import numpy as np


def independent_baseline(train, n, seed):
    """Sample each column independently (with replacement) from its empirical distribution."""
    rng = np.random.RandomState(seed)
    return train.apply(lambda s: s.iloc[rng.randint(0, len(s), n)].reset_index(drop=True))
