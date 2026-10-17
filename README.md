# Synthetic tabular data: VAE on Adult Income

Trains a plain VAE on Adult Income and evaluates synthetic rows on utility, fidelity and
nearest-neighbor privacy, against an independent-column baseline.

## Run

```bash
pip install -r requirements.txt
python run_all.py                  # full run, writes results/
python run_all.py --quick          # smoke test, writes results_quick/
python run_all.py --sdv            # also run the SDV TVAE reference (pip install sdv)
python run_all.py --data path.csv  # use a local CSV with a header row
```

Data: if `data/adult.csv` does not exist, Adult is downloaded from OpenML (version 2).
`python tests/make_fake_adult.py` writes a fake Adult-shaped file for smoke tests only.

## Iteration 2: training length and nonzero indicator

```bash
python run_length.py --quick   # smoke test
python run_length.py           # 3 seeds x 2 generator variants x checkpoints at 25/50/100/200 epochs
```

Iteration 1 (`run_all.py`) stops at the best validation loss (cap 100 epochs). Many runs hit the cap,
so iteration 2 trains to 200 epochs with no early stopping and evaluates at each checkpoint, for two
generator variants:

- `base`: same encoding as iteration 1 (`config/adult.yaml`).
- `nonzero indicator`: `config/adult_v2.yaml` adds a zero/nonzero flag for `capital-gain` and
  `capital-loss`. The value column is standardized on nonzero rows only and its loss is skipped on
  zero rows. At sampling, a zero flag forces the value to 0.

Evaluation always uses `config/adult.yaml` and the frozen privacy config, so the threshold and all
metrics are comparable with iteration 1. Validation loss is not comparable across the two variants.
Results go to `results_length/`. Report iteration 2 as a second iteration chosen after seeing
iteration-1 results (KS and detection AUC were weak); keep both tables.

## Layout

| Path | Purpose |
|---|---|
| `config/adult.yaml` | Column spec (type, transform, drop, rare threshold). No column lists in code. |
| `config/privacy.yaml` | Frozen privacy settings. Hash stored in `results/privacy_config.sha256`. |
| `config/experiment.yaml` | Seeds, VAE and XGBoost settings, latent sweep. |
| `src/preprocess.py` | Fit on train only. Invert for sampling. |
| `src/models.py` | Independent baseline, VAE, optional SDV TVAE. |
| `src/evaluate.py` | Utility, fidelity, privacy. |
| `run_all.py` | Reproduces the main table and plots. |
| `run_length.py` | Iteration 2: training-length and nonzero-indicator experiment. |
| `config/adult_v2.yaml` | Generator-side spec with nonzero indicators. |

## Decisions

- **Splits:** 70/15/15 stratified on the target, seed 0. The generator trains on train only.
  Validation loss is used for early stopping; validation rows are never trained on.
- **Preprocessing:** drop `fnlwgt` and `education`. `?` and NaN become `"missing"`. Numerics are
  standardized; `capital-gain` and `capital-loss` use `log1p` first, with no nonzero indicator.
  Categories under 0.5% frequency are grouped into `__rare__` (the target is exempt). Real and
  synthetic tables are compared after this grouping.
- **Model:** MLP encoder/decoder (2 hidden layers of 128). Gaussian head with a learned per-column
  sigma for numerics, one softmax head per categorical column. Loss is reconstruction + KL,
  with KL weight fixed at 1. Training stops at the best validation loss (patience 10, max 100 epochs).
- **Sampling:** z from the prior, decode, sample categoricals from the softmax, add Gaussian noise
  to numerics, invert preprocessing, clip to the training range, round integer columns.
- **Seeds:** 3 per configuration. Tables report mean ± std over seeds.
- **Baseline:** each column resampled independently from the training table (target included).

## Evaluation

- **Utility:** XGBoost trained on the synthetic (or real) table, ROC-AUC on the real test split.
  Ratio is synthetic AUC divided by real-train AUC for the same seed. Synthetic size is 1x train.
- **Fidelity (real train vs synthetic):** mean KS (numerics), mean TVD (categoricals), mean absolute
  difference of association matrices (Pearson for numeric pairs, Cramér's V for categorical pairs;
  mixed pairs are skipped), and a real-vs-synthetic XGBoost detection AUC (0.5 = indistinguishable).
- **Privacy:** Euclidean distance in the preprocessed space to the nearest training row.
  Validation is split into halves A and B. The threshold is the 5th percentile of half-A distances
  after dropping rows with an exact twin in train. Exact copies are detected by row identity.
  Reported per table:
  - `exact_share`: rows identical to a training row.
  - `near_share`: share of non-exact rows below the threshold. Compare it to half A's own value
    (`near_share_half_A` in `privacy_summary.json`), not to a fixed 5%: Adult distances are quantized
    (one year of age is a fixed step), so ties at the threshold pull the rate below 5%.
  - `below_share`: all rows below the threshold, exact copies included.
  - `median_ratio`: median distance divided by half A's median distance.
  - `dup_copy_share`: among exact copies, the share whose training row is duplicated in the real data.
  Half B is the control: its `near_share` should match half A's and its ratio should be near 1.
  The threshold case (twin share in half A) is written to `privacy_summary.json`.

Note on the control: Adult contains many duplicate rows once `fnlwgt` and `education` are dropped,
so a real holdout row often has an exact twin in train. That is why exact copies are reported
separately and excluded from `near_share`.

## Limitations

- Euclidean distance on one-hot columns is dominated by categorical mismatches (one mismatch adds 2
  to the squared distance, one standard deviation on a numeric adds 1).
- A `near_share` above 5% can reflect mode concentration instead of copying.
- Nearest-neighbor distance is a weak privacy test. No membership inference attack is run and there
  is no formal guarantee.
- `capital-gain` is top-coded at 99999 and about 92% zeros. In iteration 1 a Gaussian head on the
  `log1p` value smears the zero spike; iteration 2 adds a nonzero indicator.
- Rare categories and extreme values are smoothed out, which can hurt minority-class utility.
- One dataset limits generalization.
