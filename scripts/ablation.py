"""Ablation studies (Innovation D): point density and Poisson depth.

    density:  for a few models, run each method at N in {10k, 40k, 100k} on clean
              clouds; record oriented accuracy / angular error / wall-clock.
    depth:    for one model, sweep screened-Poisson depth in {8,10,12} and record
              reconstruction Chamfer / F-score.

Writes results/csv/ablation_density.csv and results/csv/ablation_depth.csv.
Point clouds at the requested densities must exist (run prepare_data.py --n ...).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import trimesh

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from scripts.run_normals import run_wnnc_treecode, run_pca_mst  # noqa: E402
from scripts.eval_normals import eval_normals  # noqa: E402
from scripts.eval_recon import eval_recon  # noqa: E402
from scripts.poisson_recon import poisson_reconstruct  # noqa: E402
from src.ours.wnnc_torch import solve_normals  # noqa: E402


def density_ablation(models, densities, methods, ours_max_n):
    rows = []
    for model in models:
        for n in densities:
            pc = f"data/pointclouds/{model}_n{n}_clean.npz"
            if not os.path.exists(pc):
                print(f"[skip] {pc} missing"); continue
            d = np.load(pc); pts, gtn = d["points"].astype(np.float64), d["gt_normals"]
            for method in methods:
                if method == "ours_torch" and n > ours_max_n:
                    print(f"[skip] ours_torch at n={n} (> ours_max_n={ours_max_n})"); continue
                if method == "wnnc_cuda":
                    nrm, t = run_wnnc_treecode(pts, "l0", 40, "float")
                elif method == "ours_torch":
                    nrm, t = solve_normals(pts, width_config="l0", iters=40,
                                           dtype=torch.float32, device="cuda")
                else:
                    nrm, t = run_pca_mst(pts, 30, 30)
                m = eval_normals(nrm, gtn)
                m.update(model=model, n_points_req=n, method=method, time_sec=t)
                rows.append(m)
                print(f"[density] {model} n={n:<7} {method:10s} "
                      f"oacc={m['oriented_acc']:.3f} ang={m['mean_ang_err_oriented']:.2f} t={t:.1f}s")
    pd.DataFrame(rows).to_csv("results/csv/ablation_density.csv", index=False)
    print("wrote results/csv/ablation_density.csv")


def depth_ablation(model, variant, method, depths):
    rows = []
    npz = f"results/normals/{model}_n40000_{variant}__{method}.npz"
    if not os.path.exists(npz):
        print(f"[skip] {npz} missing (run benchmark first)"); return
    d = np.load(npz); pts, nrm = d["points"], d["normals"]
    gt_mesh = trimesh.load(f"data/meshes/{model}.ply", process=False)
    for depth in depths:
        mesh = poisson_reconstruct(pts, nrm, depth=depth)
        out = f"results/meshes/ablation_{model}_{variant}_{method}_d{depth}.ply"
        mesh.export(out)
        rm = eval_recon(mesh, gt_mesh, n_samples=100000)
        rm.update(model=model, variant=variant, method=method, depth=depth)
        rows.append(rm)
        print(f"[depth] {model} {method} d={depth} CD_l1={rm['chamfer_l1_x1e3']:.3f} "
              f"F@1%={rm['fscore@0.01']:.3f} verts={rm['rec_verts']}")
    pd.DataFrame(rows).to_csv("results/csv/ablation_depth.csv", index=False)
    print("wrote results/csv/ablation_depth.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--density_models", nargs="+", default=["dragon", "armadillo"])
    ap.add_argument("--densities", type=int, nargs="+", default=[10000, 40000, 100000])
    ap.add_argument("--methods", nargs="+", default=["wnnc_cuda", "ours_torch", "pca_mst"])
    ap.add_argument("--ours_max_n", type=int, default=40000)
    ap.add_argument("--depth_model", default="dragon")
    ap.add_argument("--depth_variant", default="clean")
    ap.add_argument("--depth_method", default="wnnc_cuda")
    ap.add_argument("--depths", type=int, nargs="+", default=[8, 10, 12])
    ap.add_argument("--skip_density", action="store_true")
    ap.add_argument("--skip_depth", action="store_true")
    args = ap.parse_args()
    os.makedirs("results/csv", exist_ok=True)
    if not args.skip_density:
        density_ablation(args.density_models, args.densities, args.methods, args.ours_max_n)
    if not args.skip_depth:
        depth_ablation(args.depth_model, args.depth_variant, args.depth_method, args.depths)


if __name__ == "__main__":
    main()
