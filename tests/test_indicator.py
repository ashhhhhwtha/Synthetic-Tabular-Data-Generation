"""Torch-free checks for the nonzero-indicator encoding.

    python tests/make_fake_adult.py data/fake_adult.csv 6000
    python tests/test_indicator.py data/fake_adult.csv
"""
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data import load_adult, split_data
from src.preprocess import Preprocessor

root = Path(__file__).parent.parent
spec_b = yaml.safe_load(open(root / "config/adult.yaml"))
spec_g = yaml.safe_load(open(root / "config/adult_v2.yaml"))
df = load_adult(spec_b, sys.argv[1])
train, _, _ = split_data(df, spec_b["target"], 0)
base, gen = Preprocessor(spec_b).fit(train), Preprocessor(spec_g).fit(train)
train = base.group(train)

assert gen.ind == ["capital-gain", "capital-loss"] and base.ind == []
assert gen.cat_sizes == base.cat_sizes + [2, 2]
Z = gen.transform(train)
assert Z.shape[1] == gen.n_num + sum(gen.cat_sizes)
assert base.transform(train).shape[1] == Z.shape[1] - 4

# mask index points at the 'nonzero' flag of the right column
for j, c in enumerate(gen.num):
    i = gen.num_mask_idx[j]
    if c in gen.ind:
        assert (Z[:, i] == (train[c].to_numpy(float) != 0)).all(), c
        assert (Z[train[c].to_numpy(float) == 0, j] == 0).all(), "zero rows must be filled with 0"
    else:
        assert i == -1
# value column stats come from nonzero rows only
for c in gen.ind:
    j = gen.num.index(c)
    nzv = Z[train[c].to_numpy(float) != 0, j]
    assert abs(nzv.mean()) < 1e-3 and abs(nzv.std() - 1) < 1e-2, c

# round trip, including the indicator codes
off, codes = gen.n_num, []
for k in gen.cat_sizes:
    codes.append(Z[:, off:off + k].argmax(1)); off += k
back = gen.inverse(Z[:, :gen.n_num], np.stack(codes, 1))
assert list(back.columns) == list(train.columns)
assert (back[gen.cat].to_numpy() == train[gen.cat].to_numpy()).all()
assert np.allclose(back[gen.num].to_numpy(float), train[gen.num].to_numpy(float), atol=1e-3)

# a 'zero' flag forces the value to 0, and values stay in the training range
noise = np.random.RandomState(0).randn(len(train), gen.n_num) * 3
codes_arr = np.stack(codes, 1)
codes_arr[:, len(gen.cat)] = 0           # capital-gain flag = 0
b2 = gen.inverse(noise, codes_arr)
assert (b2["capital-gain"] == 0).all()
nzmask = codes_arr[:, len(gen.cat) + 1] == 1
assert (b2.loc[nzmask, "capital-loss"] >= gen.stats["capital-loss"]["lo_nz"]).all()
assert (b2["capital-loss"] <= gen.stats["capital-loss"]["hi"]).all()
print("indicator checks passed")
