"""Innovation-A validation: does our pure-PyTorch dense WNNC match the official
CUDA treecode *numerically*?  Compares two normal fields on identical points.

Both solvers start from zeros with the same b=0.5 forcing and identical iteration,
so there is no global sign ambiguity: the difference is purely the treecode's
far-field approximation error versus our exact all-pairs evaluation.
"""

from __future__ import annotations

import argparse

import numpy as np


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    dot = np.clip((a * b).sum(1), -1.0, 1.0)
    ang = np.degrees(np.arccos(dot))
    l2 = np.linalg.norm(a - b, axis=1)
    return {
        "mean_angle_deg": float(ang.mean()),
        "median_angle_deg": float(np.median(ang)),
        "p99_angle_deg": float(np.percentile(ang, 99)),
        "max_angle_deg": float(ang.max()),
        "mean_l2": float(l2.mean()),
        "median_l2": float(np.median(l2)),
        "max_l2": float(l2.max()),
        "pct_angle_lt_1deg": float((ang < 1.0).mean()),
        "agree_orientation": float((dot > 0).mean()),
        "n_points": int(len(a)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="npz A with 'normals' (e.g. ours)")
    ap.add_argument("--b", required=True, help="npz B with 'normals' (e.g. official)")
    args = ap.parse_args()
    a = np.load(args.a)["normals"]
    b = np.load(args.b)["normals"]
    assert len(a) == len(b)
    for k, v in compare(a, b).items():
        print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()
