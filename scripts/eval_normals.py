"""Normal-quality metrics (all programmatic; no human inspection).

Predicted normals live on the *same* points as the ground truth (both derive from
one input cloud), so evaluation is a direct per-point comparison -- no KNN needed.

    oriented_acc            fraction with (n_est . n_gt) > 0   (global orientation)
    mean_ang_err_oriented   mean acos(clip(n_est . n_gt))       [degrees]  (signed)
    mean_ang_err_unsigned   mean acos(|n_est . n_gt|)           [degrees]  (line-field)
    rms_ang_oriented        root-mean-square of the signed angular error
    pct_good_<k>deg         fraction of points with unsigned angular error < k deg
"""

from __future__ import annotations

import argparse

import numpy as np


def eval_normals(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12)
    gt = gt / (np.linalg.norm(gt, axis=1, keepdims=True) + 1e-12)
    dot = np.clip((pred * gt).sum(1), -1.0, 1.0)
    ang_oriented = np.degrees(np.arccos(dot))
    ang_unsigned = np.degrees(np.arccos(np.abs(dot)))
    return {
        "oriented_acc": float((dot > 0).mean()),
        "mean_ang_err_oriented": float(ang_oriented.mean()),
        "mean_ang_err_unsigned": float(ang_unsigned.mean()),
        "rms_ang_oriented": float(np.sqrt((ang_oriented ** 2).mean())),
        "pct_good_5deg": float((ang_unsigned < 5).mean()),
        "pct_good_10deg": float((ang_unsigned < 10).mean()),
        "n_points": int(len(pred)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="npz with points, normals")
    ap.add_argument("--gt", required=True, help="npz with points, gt_normals")
    args = ap.parse_args()
    pred = np.load(args.pred)["normals"]
    gt = np.load(args.gt)["gt_normals"]
    assert len(pred) == len(gt), f"count mismatch {len(pred)} vs {len(gt)}"
    m = eval_normals(pred, gt)
    for k, v in m.items():
        print(f"  {k:24s}: {v}")


if __name__ == "__main__":
    main()
