"""Spec-driven preprocessing. Everything is fitted on the training split only."""
import numpy as np
import pandas as pd

RARE = "__rare__"


class Preprocessor:
    """Numerics: optional log1p, then standardize. Categoricals: rare grouping, then one-hot.

    Layout of the encoded matrix: [numeric columns | one-hot block per categorical column |
    2-way one-hot block per nonzero-indicator column].
    The target is a categorical column and is part of the generated table.
    """

    def __init__(self, spec):
        self.spec = spec
        self.target = spec["target"]
        self.cols = {c: v for c, v in spec["columns"].items() if not v.get("drop")}
        self.num = [c for c, v in self.cols.items() if v["type"] == "numeric"]
        self.cat = [c for c, v in self.cols.items() if v["type"] == "categorical"]
        self.order = list(self.cols)
        # numeric columns with a derived zero/nonzero indicator (generator-side encoding only)
        self.ind = [c for c in self.num if self.cols[c].get("nonzero_indicator")]

    # ---- numeric helpers
    def _fwd(self, x, c):
        return np.log1p(x) if self.cols[c].get("transform") == "log1p" else x

    def _inv(self, x, c):
        return np.expm1(x) if self.cols[c].get("transform") == "log1p" else x

    # ---- fit
    def fit(self, df):
        self.stats, self.cats = {}, {}
        for c in self.num:
            raw = df[c].to_numpy(float)
            if c in self.ind:
                nz = raw[raw != 0]
                t = self._fwd(nz, c)
                lo_nz = nz.min()
            else:
                t = self._fwd(raw, c)
                lo_nz = raw.min()
            self.stats[c] = dict(mean=t.mean(), std=t.std() + 1e-8, lo=raw.min(), hi=raw.max(),
                                 lo_nz=lo_nz)
        for c in self.cat:
            min_freq = self.cols[c].get("min_freq", self.spec["rare_min_freq"])
            freq = df[c].value_counts(normalize=True)
            keep = sorted(freq[freq >= min_freq].index.tolist())
            if len(keep) < len(freq):
                keep.append(RARE)
            self.cats[c] = keep
        return self

    @property
    def n_num(self):
        return len(self.num)

    @property
    def cat_sizes(self):
        """Sizes of all one-hot blocks: real categoricals, then one 2-way block per indicator."""
        return [len(self.cats[c]) for c in self.cat] + [2] * len(self.ind)

    @property
    def num_mask_idx(self):
        """Per numeric column: index of its 'nonzero' flag in the encoded vector, or -1.
        The VAE uses this to skip the numeric loss on rows where the flag is 0."""
        start = self.n_num + sum(len(self.cats[c]) for c in self.cat)
        return [start + 2 * self.ind.index(c) + 1 if c in self.ind else -1 for c in self.num]

    # ---- transforms
    def group(self, df):
        """Map rare or unseen categories to RARE so real and synthetic share one category space."""
        df = df.copy()
        for c in self.cat:
            if RARE in self.cats[c]:
                v = df[c].astype(str).to_numpy()
                df[c] = np.where(np.isin(v, self.cats[c]), v, RARE)
        return df

    def _codes(self, s, c):
        v = s.astype(str).to_numpy()
        if RARE in self.cats[c]:
            v = np.where(np.isin(v, self.cats[c]), v, RARE)
        return pd.Categorical(v, categories=self.cats[c]).codes.astype(int)  # -1 if unseen

    def transform(self, df, include_target=True):
        blocks = []
        if self.num:
            cols = []
            for c in self.num:
                s = self.stats[c]
                raw = df[c].to_numpy(float)
                z = (self._fwd(raw, c) - s["mean"]) / s["std"]
                cols.append(np.where(raw != 0, z, 0.0) if c in self.ind else z)
            blocks.append(np.stack(cols, 1))
        for c in self.cat:
            if c == self.target and not include_target:
                continue
            codes = self._codes(df[c], c)
            oh = np.zeros((len(df), len(self.cats[c])))
            ok = codes >= 0
            oh[np.where(ok)[0], codes[ok]] = 1.0
            blocks.append(oh)
        for c in self.ind:
            oh = np.zeros((len(df), 2))
            oh[np.arange(len(df)), (df[c].to_numpy(float) != 0).astype(int)] = 1.0
            blocks.append(oh)
        return np.concatenate(blocks, 1).astype(np.float32)

    def inverse(self, num_std, cat_codes):
        """num_std: (n, n_num) standardized. cat_codes: (n, n_cat + n_ind) integer codes,
        real categoricals first, then one code (0 = zero, 1 = nonzero) per indicator column."""
        out = {}
        n_cat = len(self.cat)
        for j, c in enumerate(self.num):
            s = self.stats[c]
            lo = s["lo_nz"] if c in self.ind else s["lo"]
            t = num_std[:, j] * s["std"] + s["mean"]
            t = np.clip(t, self._fwd(lo, c), self._fwd(s["hi"], c))
            x = np.clip(self._inv(t, c), lo, s["hi"])  # training range
            if self.cols[c].get("integer"):
                x = np.round(x)
            if c in self.ind:
                x = np.where(cat_codes[:, n_cat + self.ind.index(c)] == 1, x, 0.0)
            out[c] = x
        for j, c in enumerate(self.cat):
            out[c] = np.array(self.cats[c], dtype=object)[cat_codes[:, j]]
        return pd.DataFrame(out)[self.order]

    # ---- utility features
    def xy(self, df):
        """Features (target excluded) and binary label."""
        y = (df[self.target] == self.spec["positive_label"]).astype(int).to_numpy()
        return self.transform(df, include_target=False), y
