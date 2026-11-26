"""Shared utilities: seeds, data loading, the fixed split, deterministic transforms,
normalisation, fast in-memory batching and evaluation helpers.

Every script imports from here so that all experiments use exactly the same split,
preprocessing and transforms.
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLIT_DIR = ROOT / "splits"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"
FEAT_DIR = ROOT / "features"
FIG_DIR = ROOT / "figures"

SEED = 26011  # all numeric characters of roll number M26CSA011, in order
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# Synthetic-data mode, only for smoke-testing the pipeline without CIFAR10.
FAKE = os.environ.get("A3_FAKE_DATA", "0") == "1"
N_VAL = 200 if FAKE else 5000


# --------------------------------------------------------------------------- seeds
def set_seed(seed: int):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- raw data
def load_cifar_raw():
    """Returns uint8 arrays x_train (50000,32,32,3), y_train, x_test (10000,32,32,3), y_test."""
    if FAKE:
        rng = np.random.default_rng(0)
        y_tr = np.repeat(np.arange(10), 200)
        y_te = np.repeat(np.arange(10), 50)
        # class-dependent mean colour so that the fake task is learnable
        def gen(y):
            base = (y[:, None, None, None] * 23 % 255).astype(np.float32)
            x = base + rng.normal(0, 40, size=(len(y), 32, 32, 3))
            return np.clip(x, 0, 255).astype(np.uint8)
        return gen(y_tr), y_tr, gen(y_te), y_te

    cache = DATA_DIR / "cifar10.npz"
    if cache.exists():
        d = np.load(cache)
        return d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    import torchvision
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tr = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True)
    te = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True)
    out = dict(x_train=tr.data, y_train=np.array(tr.targets, dtype=np.int64),
               x_test=te.data, y_test=np.array(te.targets, dtype=np.int64))
    np.savez(cache, **out)
    return out["x_train"], out["y_train"], out["x_test"], out["y_test"]


# --------------------------------------------------------------------------- split
def split_path():
    return SPLIT_DIR / (f"split_seed{SEED}" + ("_FAKE" if FAKE else "") + ".npz")


def make_split(y_train_full):
    idx = np.arange(len(y_train_full))
    tr_idx, va_idx = train_test_split(idx, test_size=N_VAL, stratify=y_train_full,
                                      random_state=SEED, shuffle=True)
    return np.sort(tr_idx), np.sort(va_idx)


def get_split(y_train_full):
    p = split_path()
    if p.exists():
        d = np.load(p)
        return d["train_idx"], d["val_idx"]
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    tr_idx, va_idx = make_split(y_train_full)
    np.savez(p, train_idx=tr_idx, val_idx=va_idx, seed=SEED)
    return tr_idx, va_idx


def get_data():
    """Float tensors in [0,1], NCHW, on CPU: dict split -> (x, y)."""
    xtr, ytr, xte, yte = load_cifar_raw()
    tr_idx, va_idx = get_split(ytr)
    to_t = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).float().div_(255.0)
    return {
        "train": (to_t(xtr[tr_idx]), torch.from_numpy(ytr[tr_idx]).long()),
        "val": (to_t(xtr[va_idx]), torch.from_numpy(ytr[va_idx]).long()),
        "test": (to_t(xte), torch.from_numpy(yte).long()),
    }


def stratified_subset(y, frac_or_n, seed=SEED):
    """Indices of a fixed stratified subset (fraction in (0,1] or absolute count)."""
    n = len(y)
    size = frac_or_n if frac_or_n > 1 else int(round(frac_or_n * n))
    if size >= n:
        return np.arange(n)
    idx, _ = train_test_split(np.arange(n), train_size=size, stratify=np.asarray(y),
                              random_state=seed, shuffle=True)
    return np.sort(idx)


# --------------------------------------------------------------------------- transforms
def _pixel_perm():
    return torch.from_numpy(np.random.default_rng(SEED).permutation(32 * 32))


def _block_perm():
    return torch.from_numpy(np.random.default_rng(SEED + 1).permutation(16))


def apply_transform(x, name):
    """Deterministic, image-independent transforms applied identically to train/val/test.
    x: (N,3,32,32) float in [0,1]."""
    if name in (None, "none"):
        return x
    n = x.shape[0]
    if name == "permute":  # one fixed permutation of the 1024 positions, same for all channels
        return x.reshape(n, 3, 1024)[:, :, _pixel_perm()].reshape(n, 3, 32, 32).contiguous()
    if name == "gray":     # luminance replicated to 3 channels (same architecture / params)
        g = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        return g.unsqueeze(1).repeat(1, 3, 1, 1).contiguous()
    if name == "blockshuffle":  # fixed permutation of the 16 blocks of 8x8 pixels
        b = x.reshape(n, 3, 4, 8, 4, 8).permute(0, 1, 2, 4, 3, 5).reshape(n, 3, 16, 8, 8)
        b = b[:, :, _block_perm()]
        return b.reshape(n, 3, 4, 4, 8, 8).permute(0, 1, 2, 4, 3, 5).reshape(n, 3, 32, 32).contiguous()
    raise ValueError(f"unknown transform {name}")


def channel_stats(x):
    return x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3))


def normalize(x, mean, std):
    return (x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)


def fixed_noise(shape, seed):
    """Reproducible Gaussian noise tensor (generated on CPU so it is device-independent)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g)


def add_gaussian_noise(x, sigma, noise):
    return (x + sigma * noise).clamp_(0.0, 1.0)


# --------------------------------------------------------------------------- label noise (Q1b)
def make_noisy_labels(y, frac, seed=SEED):
    """Corrupt a fixed fraction of labels; each new label is drawn uniformly from the
    OTHER 9 classes, so a corrupted label is always wrong. Returns (y_noisy, mask)."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    n = len(y)
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, size=int(round(frac * n)), replace=False)] = True
    y_noisy = y.copy()
    y_noisy[mask] = (y[mask] + rng.integers(1, 10, size=mask.sum())) % 10
    return y_noisy, mask


# --------------------------------------------------------------------------- batching / eval
class Batches:
    """Minibatches from in-memory tensors (no augmentation, so no DataLoader needed)."""

    def __init__(self, x, y, bs, shuffle, seed=SEED):
        self.x, self.y, self.bs, self.shuffle = x, y, bs, shuffle
        self.g = torch.Generator().manual_seed(seed)

    def __len__(self):
        return (len(self.x) + self.bs - 1) // self.bs

    def __iter__(self):
        n = len(self.x)
        idx = torch.randperm(n, generator=self.g) if self.shuffle else torch.arange(n)
        idx = idx.to(self.x.device)
        for i in range(0, n, self.bs):
            j = idx[i:i + self.bs]
            yield self.x[j], self.y[j]


@torch.no_grad()
def predict_logits(model, x, bs=1000):
    model.eval()
    out = [model(x[i:i + bs]).float().cpu() for i in range(0, len(x), bs)]
    return torch.cat(out)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def load_json(path):
    with open(path) as f:
        return json.load(f)
