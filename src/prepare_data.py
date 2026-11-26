"""Step 1: download CIFAR10, create and save the stratified 45k/5k split (seed 26011), sanity-check it.

Usage: python src/prepare_data.py
"""
import numpy as np

from common import CLASSES, SEED, get_split, load_cifar_raw, split_path

if __name__ == "__main__":
    xtr, ytr, xte, yte = load_cifar_raw()
    tr, va = get_split(ytr)
    assert len(np.intersect1d(tr, va)) == 0 and len(tr) + len(va) == len(ytr)
    print(f"split seed {SEED} saved to {split_path()}")
    print(f"train {len(tr)}  val {len(va)}  test {len(yte)}")
    print("val per class:", dict(zip(CLASSES, np.bincount(ytr[va], minlength=10).tolist())))
    print("train per class:", dict(zip(CLASSES, np.bincount(ytr[tr], minlength=10).tolist())))
