"""Screened Poisson surface reconstruction from an oriented point cloud (Open3D).

``create_from_point_cloud_poisson`` is Open3D's wrapper around Kazhdan &
Hoppe's *screened* Poisson reconstruction (ToG 2013) -- exactly the classical
reconstructor the assignment asks for.  Reconstruction quality becomes a
*downstream* score for normal quality: a wrong global orientation makes the
oriented indicator field inconsistent and the surface breaks, so this is a
meaningful end-to-end check of the orientation step.

We use Open3D rather than pymeshlab because the MeshLab Poisson build in this
image oversubscribes the shared host's 128 logical cores (16 allocated) and
stalls for minutes; Open3D exposes ``n_threads`` and runs the same algorithm in
~1s.  Open3D also returns per-vertex densities, letting us trim the low-density
"balloon" that Poisson extrapolates over holes in open scans.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import open3d as o3d


def poisson_reconstruct(points: np.ndarray, normals: np.ndarray, depth: int = 10,
                        density_trim_quantile: float = 0.03, n_threads: int = 8,
                        scale: float = 1.1):
    """Return a reconstructed trimesh.Trimesh via screened Poisson.

    density_trim_quantile>0 removes the lowest-density fraction of vertices,
    which cuts the extrapolated surface Poisson invents over holes (open scans).
    """
    import trimesh
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(depth), n_threads=int(n_threads), scale=scale, linear_fit=False)
    densities = np.asarray(densities)
    if density_trim_quantile > 0 and len(densities):
        thresh = np.quantile(densities, density_trim_quantile)
        mesh.remove_vertices_by_mask(densities < thresh)
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="npz with points, normals")
    ap.add_argument("--out", required=True, help="output mesh .ply")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--density_trim_quantile", type=float, default=0.03)
    ap.add_argument("--n_threads", type=int, default=8)
    args = ap.parse_args()

    d = np.load(args.pred)
    mesh = poisson_reconstruct(d["points"], d["normals"], depth=args.depth,
                               density_trim_quantile=args.density_trim_quantile,
                               n_threads=args.n_threads)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    mesh.export(args.out)
    print(f"[poisson] depth={args.depth} -> {args.out}  "
          f"({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")


if __name__ == "__main__":
    main()
