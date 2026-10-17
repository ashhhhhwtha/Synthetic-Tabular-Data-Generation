"""Second iteration: effect of training length, and of a nonzero indicator for the capital columns.

    python run_length.py --quick     # smoke test: 1 seed, checkpoints at 3 and 6 epochs
    python run_length.py             # 3 seeds x 2 generators x 4 checkpoints

Two generator variants are trained to 200 epochs with no early stopping and evaluated at each
checkpoint:
  base               config/adult.yaml     (same encoding as the main run)
  nonzero indicator  config/adult_v2.yaml  (adds a zero/nonzero flag for capital-gain/loss,
                                            numeric loss skipped on zero rows)
Evaluation always uses config/adult.yaml and the frozen privacy config, so numbers are
comparable with results/results_table.md. Validation loss is not comparable across variants.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_all import check_privacy_lock, load_yaml
from src.data import load_adult, split_data
from src.evaluate import Fidelity, PrivacyEvaluator, utility_auc
from src.models import sample_vae, train_vae_checkpoints
from src.plots import plot_length
from src.preprocess import Preprocessor

ROOT = Path(__file__).parent
METRICS = ["auc_ratio", "ks", "ks_capital-gain", "ks_capital-loss", "tvd", "assoc_diff",
           "detect_auc", "exact_share", "near_share", "median_ratio", "dup_copy_share", "val_loss"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/adult.csv"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--allow-privacy-change", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(ROOT / "config/experiment.yaml")
    spec_eval = load_yaml(ROOT / "config/adult.yaml")
    spec_gen = load_yaml(ROOT / "config/adult_v2.yaml")
    pcfg = load_yaml(ROOT / "config/privacy.yaml")
    (ROOT / "results").mkdir(exist_ok=True)
    post_hoc = check_privacy_lock(ROOT / "config/privacy.yaml", ROOT / "results",
                                  args.allow_privacy_change)
    out = ROOT / ("results_length_quick" if args.quick else "results_length")
    out.mkdir(exist_ok=True)

    seeds, ckpts, latent = cfg["seeds"], cfg["length"]["checkpoints"], cfg["length"]["latent"]
    if args.quick:
        seeds, ckpts = seeds[:1], [3, 6]

    # ---- data. Both encoders are fitted on the training split only.
    df = load_adult(spec_eval, args.data)
    train, val, test = split_data(df, spec_eval["target"], cfg["split_seed"])
    eval_pre = Preprocessor(spec_eval).fit(train)
    gens = {"base": Preprocessor(spec_eval).fit(train),
            "nonzero indicator": Preprocessor(spec_gen).fit(train)}
    train, val, test = eval_pre.group(train), eval_pre.group(val), eval_pre.group(test)

    fid = Fidelity(eval_pre, train, cfg["xgb"], cfg["detection"])
    priv = PrivacyEvaluator(eval_pre, train, val, pcfg)
    print(f"privacy threshold {priv.threshold:.4f} (main run: 0.0730); "
          f"half-A near-share {priv.near_A:.4f}")

    n = len(train)
    records = []
    for s in seeds:
        real_auc = utility_auc(eval_pre, train, test, cfg["xgb"], s)
        for name, gp in gens.items():
            t0 = time.time()
            Xtr, Xva = gp.transform(train), gp.transform(val)
            snaps = train_vae_checkpoints(Xtr, Xva, gp.n_num, gp.cat_sizes, latent, s,
                                          cfg["vae"], ckpts, gp.num_mask_idx)
            for ep, (model, val_loss) in snaps.items():
                syn = sample_vae(model, gp, n, s)
                auc = utility_auc(eval_pre, syn, test, cfg["xgb"], s)
                rec = dict(generator=name, epochs=ep, seed=s, val_loss=val_loss,
                           auc=auc, auc_ratio=auc / real_auc)
                rec.update(fid(syn, s))
                p, _ = priv.score(syn)
                rec.update(p)
                records.append(rec)
                print(f"seed {s} | {name:17s} | epochs {ep:3d} | AUC ratio {rec['auc_ratio']:.3f} "
                      f"| KS {rec['ks']:.3f} | detect {rec['detect_auc']:.3f} "
                      f"| exact {rec['exact_share']:.3f}")
            print(f"seed {s} {name} done in {time.time() - t0:.0f}s")

    res = pd.DataFrame(records)
    res["post_hoc_privacy"] = post_hoc
    res.to_csv(out / "per_seed.csv", index=False)
    cols = [m for m in METRICS if m in res.columns]
    g = res.groupby(["generator", "epochs"], sort=False)[cols]
    mean, std = g.mean(), g.std()
    table = pd.DataFrame({c: [f"{m:.3f} ± {0 if np.isnan(sd) else sd:.3f}" if not np.isnan(m) else "-"
                              for m, sd in zip(mean[c], std[c])] for c in cols}, index=mean.index)
    table.to_csv(out / "length_table.csv")
    (out / "length_table.md").write_text(table.reset_index().to_markdown(index=False))
    plot_length(res, out / "length_effect.png")
    print("\n" + table.reset_index().to_string(index=False))
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
