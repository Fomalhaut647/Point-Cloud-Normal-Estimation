"""Reconstruction-quality metrics between a reconstructed mesh and the GT mesh.

Distances are expressed as a fraction of the GT bounding-box diagonal, so the
numbers are scale-invariant and comparable across models.

    chamfer_l1   0.5*(E_rec[d] + E_gt[d])                     (x1e3, => per-mille of diag)
    chamfer_l2   0.5*(E_rec[d^2] + E_gt[d^2])                 (x1e3)
    fscore@tau   harmonic mean of precision/recall at distance threshold tau
    normal_consistency   mean |<n_rec, n_gt(nn)>| over both directions
"""

from __future__ import annotations

import argparse

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def _sample(mesh: trimesh.Trimesh, k: int, seed: int):
    pts, fid = trimesh.sample.sample_surface(mesh, k, seed=seed)
    n = mesh.face_normals[fid]
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    return np.asarray(pts), np.asarray(n)


def eval_recon(rec: trimesh.Trimesh, gt: trimesh.Trimesh, n_samples: int = 100000,
               taus=(0.005, 0.01, 0.02), seed: int = 0) -> dict:
    diag = float(np.linalg.norm(gt.vertices.max(0) - gt.vertices.min(0)))
    rp, rn = _sample(rec, n_samples, seed)
    gp, gn = _sample(gt, n_samples, seed + 1)
    rp, gp = rp / diag, gp / diag  # normalise by GT diagonal

    tg, tr = cKDTree(gp), cKDTree(rp)
    d_r2g, i_r2g = tg.query(rp)     # rec -> gt
    d_g2r, i_g2r = tr.query(gp)     # gt  -> rec

    out = {
        "chamfer_l1_x1e3": float(0.5 * (d_r2g.mean() + d_g2r.mean()) * 1e3),
        "chamfer_l2_x1e3": float(0.5 * ((d_r2g ** 2).mean() + (d_g2r ** 2).mean()) * 1e3),
    }
    for tau in taus:
        p = float((d_r2g < tau).mean())
        r = float((d_g2r < tau).mean())
        f = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
        out[f"fscore@{tau:g}"] = f
    nc_r = np.abs((rn * gn[i_r2g]).sum(1)).mean()
    nc_g = np.abs((gn * rn[i_g2r]).sum(1)).mean()
    out["normal_consistency"] = float(0.5 * (nc_r + nc_g))
    out["rec_verts"] = int(len(rec.vertices))
    out["rec_faces"] = int(len(rec.faces))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True, help="reconstructed mesh .ply")
    ap.add_argument("--gt", required=True, help="ground-truth mesh .ply")
    ap.add_argument("--n_samples", type=int, default=100000)
    args = ap.parse_args()
    rec = trimesh.load(args.rec, process=False)
    gt = trimesh.load(args.gt, process=False)
    m = eval_recon(rec, gt, n_samples=args.n_samples)
    for k, v in m.items():
        print(f"  {k:22s}: {v}")


if __name__ == "__main__":
    main()
