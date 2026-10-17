"""Utility, fidelity and privacy evaluation."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def _xgb(cfg, seed):
    return XGBClassifier(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                         learning_rate=cfg["learning_rate"], tree_method="hist",
                         n_jobs=-1, random_state=seed, eval_metric="logloss")


# ---------------------------------------------------------------- utility
def utility_auc(pre, train_df, test_df, xgb_cfg, seed):
    """Train XGBoost on `train_df` (real or synthetic), return ROC-AUC on the real test split."""
    X, y = pre.xy(train_df)
    Xte, yte = pre.xy(test_df)
    if len(np.unique(y)) < 2:
        return 0.5
    clf = _xgb(xgb_cfg, seed).fit(X, y)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


# ---------------------------------------------------------------- fidelity
def cramers_v(a, b):
    ia, _ = pd.factorize(a)
    ib, _ = pd.factorize(b)
    ka, kb = ia.max() + 1, ib.max() + 1
    ct = np.bincount(ia * kb + ib, minlength=ka * kb).reshape(ka, kb).astype(float)
    n = ct.sum()
    exp = np.outer(ct.sum(1), ct.sum(0)) / n
    chi2 = ((ct - exp) ** 2 / exp).sum()
    k = min(ka, kb) - 1
    return 0.0 if k == 0 else float(np.sqrt(chi2 / (n * k)))


def assoc_matrix(df, pre):
    """Pearson for numeric pairs, Cramer's V for categorical pairs, NaN for mixed pairs."""
    cols = pre.num + pre.cat
    nn_ = len(pre.num)
    M = np.full((len(cols), len(cols)), np.nan)
    X = df[pre.num].to_numpy(float) if pre.num else None
    for i in range(len(cols)):
        M[i, i] = 1.0
        for j in range(i + 1, len(cols)):
            if i < nn_ and j < nn_:
                a, b = X[:, i], X[:, j]
                v = 0.0 if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])
            elif i >= nn_ and j >= nn_:
                v = cramers_v(df[cols[i]].to_numpy(), df[cols[j]].to_numpy())
            else:
                continue
            M[i, j] = M[j, i] = v
    return M


def tvd(p_series, q_series):
    p = p_series.value_counts(normalize=True)
    q = q_series.value_counts(normalize=True)
    return 0.5 * p.sub(q, fill_value=0).abs().sum()


class Fidelity:
    """Fidelity of a table against the real training table (real association matrix cached)."""

    def __init__(self, pre, real, xgb_cfg, detection=True):
        self.pre, self.real, self.xgb_cfg, self.detection = pre, real, xgb_cfg, detection
        self.real_assoc = assoc_matrix(real, pre)

    def __call__(self, syn, seed):
        pre = self.pre
        ks_cols = {c: ks_2samp(self.real[c], syn[c]).statistic for c in pre.num}
        ks = np.mean(list(ks_cols.values()))
        tv = np.mean([tvd(self.real[c], syn[c]) for c in pre.cat])
        diff = np.abs(self.real_assoc - assoc_matrix(syn, pre))
        res = dict(ks=ks, tvd=tv, assoc_diff=np.nanmean(diff[np.triu_indices_from(diff, 1)]))
        res.update({f"ks_{c}": v for c, v in ks_cols.items()})
        if self.detection:
            res["detect_auc"] = self.detect(syn, seed)
        return res

    def detect(self, syn, seed):
        """Classifier tries to separate real from synthetic rows. 0.5 = indistinguishable."""
        m = min(len(self.real), len(syn), 20000)
        a = self.pre.transform(self.real.sample(m, random_state=seed))
        b = self.pre.transform(syn.sample(m, random_state=seed))
        X, y = np.vstack([a, b]), np.r_[np.zeros(m), np.ones(m)]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=seed)
        clf = _xgb(dict(self.xgb_cfg, n_estimators=100), seed).fit(Xtr, ytr)
        return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


# ---------------------------------------------------------------- privacy
def nn_dist(Q, R, chunk=1000):
    """Euclidean distance from each row of Q to its nearest row of R (float32, chunked)."""
    Q, R = Q.astype(np.float32), R.astype(np.float32)
    r2 = (R ** 2).sum(1)
    out = []
    for i in range(0, len(Q), chunk):
        q = Q[i:i + chunk]
        d2 = (q ** 2).sum(1, keepdims=True) + r2[None] - 2 * q @ R.T
        out.append(np.sqrt(np.maximum(d2.min(1), 0)))
    return np.concatenate(out)


def row_keys(Z):
    return [r.tobytes() for r in Z]


class PrivacyEvaluator:
    """Nearest-neighbor distance test in the preprocessed space (frozen config).

    Validation is split into halves A and B. Half A sets the threshold, half B is a control.
    Exact copies are detected by row identity, not by float distance.
    """

    def __init__(self, pre, train, val, cfg):
        self.pre, self.cfg = pre, cfg
        self.Ztr = pre.transform(train)
        self.counts = {}
        for k in row_keys(self.Ztr):
            self.counts[k] = self.counts.get(k, 0) + 1
        idx = np.random.RandomState(cfg["split_seed"]).permutation(len(val))
        half = len(val) // 2
        self.A, self.B = val.iloc[idx[:half]], val.iloc[idx[half:]]
        self.dA, _ = self._dist(pre.transform(self.A))
        self.zero_share_A = float((self.dA == 0).mean())
        ref = self.dA[self.dA > 0] if cfg.get("drop_exact_twins", True) else self.dA
        self.threshold = float(np.quantile(ref, cfg["quantile"]))
        self.median_A = float(np.median(self.dA))
        # Half A's own near-share under the same rule. Equals ~5% only if distances have no ties;
        # Adult distances are quantized (e.g. one year of age), so compare B and synthetic to this.
        nz = self.dA > 0
        self.near_A = float(((self.dA > 0) & (self.dA < self.threshold)).sum() / max(nz.sum(), 1))

    def _dist(self, Z):
        d = nn_dist(Z, self.Ztr)
        cnt = np.array([self.counts.get(k, 0) for k in row_keys(Z)])
        return np.where(cnt > 0, 0.0, d), cnt

    def score(self, df):
        """Returns metrics and the raw distances for a table."""
        d, cnt = self._dist(self.pre.transform(df))
        exact = d == 0
        near = (d > 0) & (d < self.threshold)
        res = dict(
            exact_share=exact.mean(),                       # rows identical to a train row
            near_share=near.sum() / max((~exact).sum(), 1),  # non-exact rows below threshold
            below_share=(d < self.threshold).mean(),         # all rows below threshold
            median_ratio=np.median(d) / self.median_A if self.median_A > 0 else np.nan,
            dup_copy_share=(cnt[exact] > 1).mean() if exact.any() else np.nan,
        )
        return res, d
