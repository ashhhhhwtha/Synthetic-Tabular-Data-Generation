"""Generators: a plain VAE and an optional SDV TVAE reference."""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- VAE
class VAE(nn.Module):
    """MLP encoder/decoder. Gaussian head (learned per-column sigma) for numerics,
    one softmax head per categorical column."""

    def __init__(self, n_num, cat_sizes, latent=32, hidden=128, mask_idx=None):
        super().__init__()
        self.n_num, self.cat_sizes, self.latent = n_num, list(cat_sizes), latent
        # mask_idx[j] >= 0: skip numeric column j's loss on rows where x[:, mask_idx[j]] == 0
        self.mask_idx = list(mask_idx) if mask_idx is not None else [-1] * n_num
        d = n_num + sum(cat_sizes)
        self.enc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent)
        self.lv = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, d))
        self.log_sigma = nn.Parameter(torch.zeros(n_num))

    def sigma(self):
        return torch.exp(self.log_sigma.clamp(-5, 2))

    def loss_terms(self, x):
        h = self.enc(x)
        mu, lv = self.mu(h), self.lv(h).clamp(-10, 10)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        out = self.dec(z)
        sig = self.sigma()
        nll = 0.5 * ((x[:, :self.n_num] - out[:, :self.n_num]) / sig) ** 2 + torch.log(sig)
        if any(i >= 0 for i in self.mask_idx):
            w = torch.ones_like(nll)
            for j, i in enumerate(self.mask_idx):
                if i >= 0:
                    w[:, j] = x[:, i]
            nll = nll * w
        rec_num = nll.sum(1)
        rec_cat, off = 0.0, self.n_num
        for k in self.cat_sizes:
            rec_cat = rec_cat + F.cross_entropy(out[:, off:off + k], x[:, off:off + k].argmax(1),
                                                reduction="none")
            off += k
        kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)
        return rec_num, rec_cat, kl


def train_vae(Xtr, Xva, n_num, cat_sizes, latent, seed, cfg, max_epochs=None, mask_idx=None):
    """Train with Adam; keep the epoch with the lowest validation loss (early stopping)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = VAE(n_num, cat_sizes, latent, cfg["hidden"], mask_idx)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    Xtr, Xva = torch.from_numpy(Xtr), torch.from_numpy(Xva)
    beta, bs = cfg["beta"], cfg["batch_size"]
    best, best_state, best_ep, bad = float("inf"), None, 0, 0
    epochs = max_epochs or cfg["max_epochs"]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            r1, r2, kl = model.loss_terms(Xtr[perm[i:i + bs]])
            loss = (r1 + r2 + beta * kl).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            r1, r2, kl = model.loss_terms(Xva)
            val = (r1 + r2 + beta * kl).mean().item()
        if val < best - 1e-4:
            best, best_state, best_ep, bad = val, copy.deepcopy(model.state_dict()), ep + 1, 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, dict(val_loss=best, best_epoch=best_ep, epochs_run=ep + 1)


def train_vae_checkpoints(Xtr, Xva, n_num, cat_sizes, latent, seed, cfg, checkpoints, mask_idx=None):
    """Train for max(checkpoints) epochs with no early stopping and keep a copy of the model at
    each checkpoint epoch. Returns {epoch: (model, validation_loss)}."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = VAE(n_num, cat_sizes, latent, cfg["hidden"], mask_idx)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    Xtr, Xva = torch.from_numpy(Xtr), torch.from_numpy(Xva)
    beta, bs = cfg["beta"], cfg["batch_size"]
    snaps = {}
    for ep in range(1, max(checkpoints) + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            r1, r2, kl = model.loss_terms(Xtr[perm[i:i + bs]])
            loss = (r1 + r2 + beta * kl).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep in checkpoints:
            model.eval()
            with torch.no_grad():
                r1, r2, kl = model.loss_terms(Xva)
                val = (r1 + r2 + beta * kl).mean().item()
            snaps[ep] = (copy.deepcopy(model), val)
    return snaps


@torch.no_grad()
def sample_vae(model, pre, n, seed):
    """Draw z from the prior, decode, sample categoricals, invert preprocessing."""
    g = torch.Generator().manual_seed(seed)
    out = model.dec(torch.randn(n, model.latent, generator=g))
    num = out[:, :model.n_num] + torch.randn(n, model.n_num, generator=g) * model.sigma()
    codes, off = [], model.n_num
    for k in model.cat_sizes:
        p = F.softmax(out[:, off:off + k], 1)
        codes.append(torch.multinomial(p, 1, generator=g).squeeze(1))
        off += k
    codes = torch.stack(codes, 1).numpy() if codes else np.zeros((n, 0), int)
    return pre.inverse(num.numpy(), codes)


# ---------------------------------------------------------------- SDV reference
def sdv_tvae(train, n, seed, epochs):
    """Optional reference. Requires `pip install sdv`."""
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import TVAESynthesizer
    md = SingleTableMetadata()
    md.detect_from_dataframe(train)
    torch.manual_seed(seed)
    np.random.seed(seed)
    synth = TVAESynthesizer(md, epochs=epochs)
    synth.fit(train)
    return synth.sample(n)[train.columns].reset_index(drop=True)
