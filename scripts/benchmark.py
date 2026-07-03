"""End-to-end benchmark orchestrator.

Runs the full matrix  {models} x {noise variants} x {methods}, computing normal
metrics for every cell and (optionally) screened-Poisson reconstruction + mesh
metrics.  Everything is cached (skips work whose output npz already exists unless
--force) and written to tidy CSVs for downstream tables/figures.

    results/csv/normals.csv   one row per (model, variant, method)
    results/csv/recon.csv     one row per (model, variant, method) reconstruction
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from scripts.run_normals import run_wnnc_treecode, run_pca_mst  # noqa: E402
from scripts.eval_normals import eval_normals  # noqa: E402
from scripts.eval_recon import eval_recon  # noqa: E402
from scripts.poisson_recon import poisson_reconstruct  # noqa: E402
from src.ours.wnnc_torch import solve_normals  # noqa: E402

# map noise variant -> WNNC smoothing-width preset (per the WNNC paper's guidance)
WIDTH_FOR = {"clean": "l0", "sigmap0025": "l2", "sigmap005": "l3", "sigmap01": "l4"}


def compute_normals(method, pts, width_config):
    if method == "wnnc_cuda":
        return run_wnnc_treecode(pts, width_config, iters=40, dtype_str="float")
    if method == "ours_torch":
        return solve_normals(pts, width_config=width_config, iters=40,
                             dtype=torch.float32, device="cuda")
    if method == "pca_mst":
        return run_pca_mst(pts, knn=30, orient_k=30)
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["sphere", "torus", "bunny", "armadillo", "dragon"])
    ap.add_argument("--variants", nargs="+",
                    default=["clean", "sigmap0025", "sigmap005", "sigmap01"])
    ap.add_argument("--methods", nargs="+",
                    default=["wnnc_cuda", "ours_torch", "pca_mst"])
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--do_recon", action="store_true")
    ap.add_argument("--poisson_depth", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs("results/normals", exist_ok=True)
    os.makedirs("results/meshes", exist_ok=True)
    os.makedirs("results/csv", exist_ok=True)

    nrows, rrows = [], []
    for model in args.models:
        gt_mesh_path = f"data/meshes/{model}.ply"
        for variant in args.variants:
            pc_path = f"data/pointclouds/{model}_n{args.n}_{variant}.npz"
            if not os.path.exists(pc_path):
                print(f"[skip] missing {pc_path}")
                continue
            d = np.load(pc_path)
            pts, gtn = d["points"].astype(np.float64), d["gt_normals"]
            width_config = WIDTH_FOR.get(variant, "l0")
            for method in args.methods:
                tag = f"{model}_n{args.n}_{variant}"
                npz = f"results/normals/{tag}__{method}.npz"
                if os.path.exists(npz) and not args.force:
                    normals = np.load(npz)["normals"]
                    elapsed = float("nan")
                else:
                    normals, elapsed = compute_normals(method, pts, width_config)
                    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
                    np.savez(npz, points=pts.astype(np.float32), normals=normals.astype(np.float32))
                m = eval_normals(normals, gtn)
                m.update(model=model, variant=variant, method=method,
                         width_config=(width_config if method != "pca_mst" else "-"),
                         time_sec=elapsed)
                nrows.append(m)
                print(f"[normals] {tag:32s} {method:10s} "
                      f"oacc={m['oriented_acc']:.3f} ang={m['mean_ang_err_oriented']:.2f} "
                      f"t={elapsed if elapsed==elapsed else -1:.1f}s")

                if args.do_recon:
                    mesh_path = f"results/meshes/{tag}__{method}_d{args.poisson_depth}.ply"
                    try:
                        if not os.path.exists(mesh_path) or args.force:
                            t0 = time.time()
                            mesh = poisson_reconstruct(normals=normals, points=pts,
                                                       depth=args.poisson_depth)
                            mesh.export(mesh_path)
                            rtime = time.time() - t0
                        else:
                            import trimesh
                            mesh = trimesh.load(mesh_path, process=False)
                            rtime = float("nan")
                        import trimesh
                        gt_mesh = trimesh.load(gt_mesh_path, process=False)
                        rm = eval_recon(mesh, gt_mesh, n_samples=100000)
                        rm.update(model=model, variant=variant, method=method,
                                  depth=args.poisson_depth, poisson_time_sec=rtime)
                        rrows.append(rm)
                        print(f"  [recon] {method:10s} CD_l1={rm['chamfer_l1_x1e3']:.3f} "
                              f"NC={rm['normal_consistency']:.3f}")
                    except Exception as e:
                        print(f"  [recon] {method} FAILED: {e!r}")

    pd.DataFrame(nrows).to_csv("results/csv/normals.csv", index=False)
    print(f"\nwrote results/csv/normals.csv ({len(nrows)} rows)")
    if rrows:
        pd.DataFrame(rrows).to_csv("results/csv/recon.csv", index=False)
        print(f"wrote results/csv/recon.csv ({len(rrows)} rows)")


if __name__ == "__main__":
    main()
