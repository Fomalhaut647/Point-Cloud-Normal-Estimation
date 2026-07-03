"""Unified normal-estimation runner for all three benchmarked methods.

    method = wnnc_cuda   official WNNC treecode (jsnln/WNNC, CUDA extension)
    method = ours_torch  our pure-PyTorch dense WNNC (src/ours/wnnc_torch.py)
    method = pca_mst     classic Hoppe'92: local PCA + MST orientation (Open3D)

All methods share the *same* preprocessing so results are directly comparable.
Output: results/normals/<tag>__<method>.npz  with keys
    points   (N,3)  original (un-normalised) coordinates
    normals  (N,3)  estimated unit normals (globally oriented)
and a sibling .json with timing / parameters.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from time import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ours.wnnc_torch import PRESET_WIDTHS, normalize_points, solve_normals  # noqa: E402


def load_points(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".npz":
        return np.load(path)["points"][:, :3].astype(np.float64)
    if ext == ".xyz":
        return np.loadtxt(path)[:, :3].astype(np.float64)
    if ext == ".npy":
        return np.load(path)[:, :3].astype(np.float64)
    if ext in (".ply", ".obj"):
        import trimesh
        return np.asarray(trimesh.load(path, process=False).vertices, dtype=np.float64)
    raise ValueError(ext)


def run_wnnc_treecode(pts: np.ndarray, width_config: str, iters: int, dtype_str: str):
    """Official WNNC treecode loop (a faithful copy of main_wnnc.py's core)."""
    import wn_treecode
    dtype = torch.float32 if dtype_str == "float" else torch.float64
    pts_n, _, _ = normalize_points(pts)
    points = torch.from_numpy(pts_n).contiguous().to(dtype).cuda()
    normals = torch.zeros_like(points)
    b = torch.ones(points.shape[0], 1, dtype=dtype, device="cuda") * 0.5
    widths = torch.ones_like(points[:, 0])
    wsmin, wsmax = PRESET_WIDTHS[width_config]
    wn = wn_treecode.WindingNumberTreecode(points)
    torch.cuda.synchronize()
    t0 = time()
    with torch.no_grad():
        out_normals = normals
        for i in range(iters):
            ws = wsmin + ((iters - 1 - i) / (iters - 1)) * (wsmax - wsmin)
            A_mu = wn.forward_A(normals, widths * ws)
            AT_A_mu = wn.forward_AT(A_mu, widths * ws)
            r = wn.forward_AT(b, widths * ws) - AT_A_mu
            A_r = wn.forward_A(r, widths * ws)
            alpha = (r * r).sum() / (A_r * A_r).sum()
            normals = normals + alpha * r
            out_normals = wn.forward_G(normals, widths * ws)
            out_normals = F.normalize(out_normals, dim=-1).contiguous()
            normals_len = torch.linalg.norm(normals, dim=-1, keepdim=True)
            normals = out_normals.clone() * normals_len
    torch.cuda.synchronize()
    return out_normals.detach().cpu().numpy(), time() - t0


def run_pca_mst(pts: np.ndarray, knn: int, orient_k: int):
    """Hoppe 1992: local PCA normals + consistent orientation via MST (Open3D)."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    t0 = time()
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    pcd.orient_normals_consistent_tangent_plane(k=orient_k)
    elapsed = time() - t0
    return np.asarray(pcd.normals), elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="input point cloud (.npz/.xyz/.npy/.ply)")
    ap.add_argument("--method", required=True, choices=["wnnc_cuda", "ours_torch", "pca_mst"])
    ap.add_argument("--tag", default=None, help="output basename tag; defaults to input stem")
    ap.add_argument("--out_dir", default="results/normals")
    ap.add_argument("--width_config", default="l0", choices=list(PRESET_WIDTHS))
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--dtype", default="float", choices=["float", "double"])
    ap.add_argument("--block_pairs", type=int, default=120_000_000)
    ap.add_argument("--knn", type=int, default=30, help="pca_mst: neighbours for PCA")
    ap.add_argument("--orient_k", type=int, default=30, help="pca_mst: neighbours for MST orient")
    args = ap.parse_args()

    pts = load_points(args.input)
    tag = args.tag or os.path.basename(args.input).rsplit(".", 1)[0]
    os.makedirs(args.out_dir, exist_ok=True)

    if args.method == "wnnc_cuda":
        normals, elapsed = run_wnnc_treecode(pts, args.width_config, args.iters, args.dtype)
        params = dict(width_config=args.width_config, iters=args.iters, dtype=args.dtype)
    elif args.method == "ours_torch":
        normals, elapsed = solve_normals(
            pts, width_config=args.width_config, iters=args.iters,
            dtype=(torch.float32 if args.dtype == "float" else torch.float64),
            device="cuda", block_pairs=args.block_pairs)
        params = dict(width_config=args.width_config, iters=args.iters,
                      dtype=args.dtype, block_pairs=args.block_pairs)
    else:  # pca_mst
        normals, elapsed = run_pca_mst(pts, args.knn, args.orient_k)
        params = dict(knn=args.knn, orient_k=args.orient_k)

    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    dst = os.path.join(args.out_dir, f"{tag}__{args.method}.npz")
    np.savez(dst, points=pts.astype(np.float32), normals=normals.astype(np.float32))
    meta = dict(method=args.method, tag=tag, n_points=int(len(pts)),
                time_sec=float(elapsed), params=params, input=args.input)
    with open(dst.replace(".npz", ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{args.method}] {tag}: {len(pts)} pts, {elapsed:.3f}s -> {dst}")


if __name__ == "__main__":
    main()
