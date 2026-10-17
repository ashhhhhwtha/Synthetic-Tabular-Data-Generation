"""Reproduce the main results table.

    python run_all.py                      # full run (3 seeds, latent sweep 8/32/128)
    python run_all.py --quick              # smoke test (1 seed, 1 latent, 15 epochs)
    python run_all.py --sdv                # add SDV TVAE reference (pip install sdv)
    python run_all.py --data data/adult.csv
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data import load_adult, split_data
from src.evaluate import Fidelity, PrivacyEvaluator, utility_auc
from src.baseline import independent_baseline
from src.models import sample_vae, sdv_tvae, train_vae
from src.plots import plot_assoc, plot_distributions, plot_nn
from src.preprocess import Preprocessor

ROOT = Path(__file__).parent
METRICS = ["auc", "auc_ratio", "ks", "tvd", "assoc_diff", "detect_auc",
           "exact_share", "near_share", "below_share", "median_ratio", "dup_copy_share"]


def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def check_privacy_lock(path, out_dir, allow_change):
    """Freeze the privacy config: store its hash on first run, fail if it changes later."""
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    lock = out_dir / "privacy_config.sha256"
    if not lock.exists():
        lock.write_text(h)
        return False
    if lock.read_text().strip() != h:
        if not allow_change:
            raise SystemExit("privacy.yaml changed since it was frozen. Revert it, or rerun with "
                             "--allow-privacy-change (results must then be labeled post hoc).")
        return True
    return False


def evaluate_one(name, syn, size_syn, seed, ctx, extra=None):
    """Utility at every size, fidelity + privacy at 1x. Returns (record, privacy distances)."""
    pre, test, cfg = ctx["pre"], ctx["test"], ctx["cfg"]
    rec = dict(method=name, seed=seed, size=1.0, **(extra or {}))
    rec["auc"] = utility_auc(pre, syn, test, cfg["xgb"], seed)
    rec["auc_ratio"] = rec["auc"] / ctx["real_auc"][seed][1.0]
    rec.update(ctx["fid"](syn, seed))
    priv, dist = ctx["priv"].score(syn)
    rec.update(priv)
    return rec, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/adult.csv"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sdv", action="store_true")
    ap.add_argument("--allow-privacy-change", action="store_true")
    args = ap.parse_args()

    out = ROOT / ("results_quick" if args.quick else "results")
    out.mkdir(exist_ok=True)
    cfg = load_yaml(ROOT / "config/experiment.yaml")
    spec = load_yaml(ROOT / "config/adult.yaml")
    pcfg = load_yaml(ROOT / "config/privacy.yaml")
    post_hoc = check_privacy_lock(ROOT / "config/privacy.yaml", out, args.allow_privacy_change)
    assert 1.0 in cfg["sizes"], "sizes must include 1.0"
    if args.quick:
        cfg["seeds"], cfg["vae"]["latent_sweep"] = cfg["seeds"][:1], [cfg["vae"]["main_latent"]]
        cfg["vae"]["max_epochs"] = 15

    # ---- data
    df = load_adult(spec, args.data)
    train, val, test = split_data(df, spec["target"], cfg["split_seed"])
    pre = Preprocessor(spec).fit(train)
    train, val, test = pre.group(train), pre.group(val), pre.group(test)
    Xtr, Xva = pre.transform(train), pre.transform(val)
    print(f"rows: train {len(train)}, val {len(val)}, test {len(test)}; encoded dim {Xtr.shape[1]}")

    # ---- evaluators (privacy settings come from the frozen config)
    fid = Fidelity(pre, train, cfg["xgb"], cfg["detection"])
    priv = PrivacyEvaluator(pre, train, val, pcfg)
    print(f"privacy threshold {priv.threshold:.4f}; half-A exact-twin share {priv.zero_share_A:.3f}")

    # ---- real references
    real_auc = {s: {} for s in cfg["seeds"]}
    for s in cfg["seeds"]:
        for size in cfg["sizes"]:
            sub = train.sample(frac=min(size, 1.0), random_state=s) if size < 1 else train
            real_auc[s][size] = utility_auc(pre, sub, test, cfg["xgb"], s)
    ctx = dict(pre=pre, test=test, cfg=cfg, fid=fid, priv=priv, real_auc=real_auc)

    records, dists, samples = [], {}, {}
    for s in cfg["seeds"]:
        records.append(dict(method="Real (train)", seed=s, size=1.0, auc=real_auc[s][1.0],
                            auc_ratio=1.0))
        # control: half-B real rows (held out, never seen by the generator)
        rec = dict(method="Real holdout (half B)", seed=s, size=1.0)
        rec.update(fid(priv.B, s))
        p, d = priv.score(priv.B)
        rec.update(p)
        records.append(rec)
        if s == cfg["seeds"][0]:
            dists["real holdout B"] = (d, p["exact_share"])
            dists["real holdout A"] = (priv.dA, priv.zero_share_A)

    n = len(train)
    for s in cfg["seeds"]:
        t0 = time.time()
        syn = independent_baseline(train, n, s)
        rec, d = evaluate_one("Independent columns", syn, n, s, ctx)
        records.append(rec)
        if s == cfg["seeds"][0]:
            dists["independent"], samples["Independent columns"] = (d, rec["exact_share"]), syn

        for latent in cfg["vae"]["latent_sweep"]:
            model, info = train_vae(Xtr, Xva, pre.n_num, pre.cat_sizes, latent, s, cfg["vae"])
            syn = sample_vae(model, pre, n, s)
            rec, d = evaluate_one(f"VAE (latent={latent})", syn, n, s, ctx, info)
            rec["latent"] = latent
            # utility vs synthetic size (fidelity/privacy stay at 1x)
            for size in cfg["sizes"]:
                if size == 1.0:
                    continue
                bigger = sample_vae(model, pre, int(n * size), s + 1000)
                a = utility_auc(pre, bigger, test, cfg["xgb"], s)
                records.append(dict(method=f"VAE (latent={latent})", seed=s, size=size,
                                    latent=latent, auc=a, auc_ratio=a / real_auc[s][size]))
            records.append(rec)
            if latent == cfg["vae"]["main_latent"] and s == cfg["seeds"][0]:
                dists["VAE"] = (d, rec["exact_share"])
                samples["VAE"] = syn
            print(f"seed {s} latent {latent}: AUC ratio {rec['auc_ratio']:.3f}, "
                  f"best epoch {info['best_epoch']}, exact {rec['exact_share']:.3f}")

        if args.sdv:
            syn = sdv_tvae(train, n, s, cfg["sdv"]["epochs"])
            rec, d = evaluate_one("SDV TVAE (reference)", syn, n, s, ctx)
            records.append(rec)
        print(f"seed {s} done in {time.time() - t0:.0f}s")

    # ---- results
    res = pd.DataFrame(records)
    res["post_hoc_privacy"] = post_hoc
    res.to_csv(out / "per_seed.csv", index=False)
    cols = [m for m in METRICS if m in res.columns]
    g = res.groupby(["method", "size"], sort=False)[cols]
    mean, std = g.mean(), g.std()
    table = pd.DataFrame({c: [f"{m:.3f} ± {0 if np.isnan(sd) else sd:.3f}" if not np.isnan(m) else "-"
                              for m, sd in zip(mean[c], std[c])] for c in cols}, index=mean.index)
    table.to_csv(out / "results_table.csv")
    (out / "results_table.md").write_text(table.reset_index().to_markdown(index=False))
    meta = dict(threshold=priv.threshold, zero_share_half_A=priv.zero_share_A,
                threshold_case="exact twins dropped" if pcfg.get("drop_exact_twins", True)
                else "all half-A rows",
                post_hoc_privacy=post_hoc, seeds=cfg["seeds"])
    (out / "privacy_summary.json").write_text(json.dumps(meta, indent=2))
    print("\n" + table.reset_index().to_string(index=False))

    # ---- plots (seed 0, main latent)
    if "VAE" in samples:
        plot_distributions(train, samples["VAE"], pre, out / "distributions.png")
        plot_assoc(train, samples["VAE"], pre, out / "assoc_heatmaps.png")
    plot_nn(dists, priv.threshold, out / "nn_distances.png")
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
