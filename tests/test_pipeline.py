"""Torch-free sanity checks: preprocessing round trip, baseline, fidelity, privacy, plots.

    python tests/make_fake_adult.py data/fake_adult.csv 6000
    python tests/test_pipeline.py data/fake_adult.csv
"""
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.baseline import independent_baseline
from src.data import load_adult, split_data
from src.evaluate import Fidelity, PrivacyEvaluator, utility_auc
from src.plots import plot_assoc, plot_distributions, plot_nn
from src.preprocess import Preprocessor

root = Path(__file__).parent.parent
spec = yaml.safe_load(open(root / "config/adult.yaml"))
pcfg = yaml.safe_load(open(root / "config/privacy.yaml"))
xgb = yaml.safe_load(open(root / "config/experiment.yaml"))["xgb"]

df = load_adult(spec, sys.argv[1])
train, val, test = split_data(df, spec["target"], 0)
pre = Preprocessor(spec).fit(train)
train, val, test = pre.group(train), pre.group(val), pre.group(test)

# 1. round trip: transform then inverse reproduces the table
Z = pre.transform(train)
off = pre.n_num
codes = []
for k in pre.cat_sizes:
    codes.append(Z[:, off:off + k].argmax(1)); off += k
back = pre.inverse(Z[:, :pre.n_num], np.stack(codes, 1))
assert (back[pre.cat].to_numpy() == train[pre.cat].to_numpy()).all(), "categorical round trip"
assert np.allclose(back[pre.num].to_numpy(float), train[pre.num].to_numpy(float), atol=1e-3), "numeric round trip"
assert Z.shape[1] == pre.n_num + sum(pre.cat_sizes)
print("round trip ok; encoded dim", Z.shape[1])

# 2. evaluators
fid = Fidelity(pre, train, xgb, True)
priv = PrivacyEvaluator(pre, train, val, pcfg)
print(f"threshold {priv.threshold:.3f}, half-A twin share {priv.zero_share_A:.3f}")
real_auc = utility_auc(pre, train, test, xgb, 0)

base = independent_baseline(train, len(train), 0)
assert base.shape == train.shape and list(base.columns) == list(train.columns)
copy = train.sample(len(train), random_state=0).reset_index(drop=True)   # 100% copies

for name, t in [("real half B", priv.B), ("independent", base), ("copy of train", copy)]:
    p, d = priv.score(t)
    f = fid(t, 0)
    auc = utility_auc(pre, t, test, xgb, 0) if len(t) > 2000 else float("nan")
    print(f"{name:14s} auc_ratio {auc/real_auc:.3f} | ks {f['ks']:.3f} tvd {f['tvd']:.3f} "
          f"assoc {f['assoc_diff']:.3f} detect {f['detect_auc']:.3f} | "
          f"exact {p['exact_share']:.3f} near {p['near_share']:.3f} below {p['below_share']:.3f} "
          f"ratio {p['median_ratio']:.2f}")

pb, _ = priv.score(priv.B)
assert 0.02 < pb["near_share"] < 0.10, "control near_share should be about 5%"
pc, _ = priv.score(copy)
assert pc["exact_share"] > 0.99, "copies must be detected as exact"

# 3. plots
out = root / "results_quick"; out.mkdir(exist_ok=True)
plot_distributions(train, base, pre, out / "test_dist.png", "baseline")
plot_assoc(train, base, pre, out / "test_assoc.png", "baseline")
(_, dB) = priv.score(priv.B); (_, dS) = priv.score(base)
plot_nn({"half B": (dB, pb["exact_share"]), "baseline": (dS, 0.0)}, priv.threshold, out / "test_nn.png")
print("all checks passed")
