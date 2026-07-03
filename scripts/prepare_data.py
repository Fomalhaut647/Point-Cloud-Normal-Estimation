"""Build the benchmark dataset: ground-truth meshes -> sampled point clouds.

For every model we obtain a triangle mesh with *consistent* face orientation,
then draw ``N`` surface samples; each sample inherits its triangle's face normal
as the **ground-truth oriented normal** (the golden standard for evaluation).
Gaussian noise (std = sigma * bbox-diagonal) yields noisy variants; the GT normal
of a noisy point is the underlying clean-surface normal.

Models
------
synthetic (exact, watertight):  sphere, torus
stanford scans (GT = mesh face normals):  bunny, armadillo, dragon, happy

Outputs
-------
data/meshes/<model>.ply                          GT mesh (for reconstruction eval)
data/pointclouds/<model>_n<N>_clean.npz          points(N,3), gt_normals(N,3)
data/pointclouds/<model>_n<N>_sigma<p>.npz       noisy variants
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import trimesh

RAW = "data/meshes_raw"
MESH_DIR = "data/meshes"
PC_DIR = "data/pointclouds"

STANFORD = {
    "bunny": f"{RAW}/bunny/reconstruction/bun_zipper.ply",
    "armadillo": f"{RAW}/Armadillo.ply",
    "dragon": f"{RAW}/dragon_recon/dragon_vrip.ply",
    "happy": f"{RAW}/happy_recon/happy_vrip.ply",
}


def make_sphere() -> trimesh.Trimesh:
    m = trimesh.creation.icosphere(subdivisions=6, radius=1.0)
    trimesh.repair.fix_normals(m)
    return m


def make_torus(R: float = 1.0, r: float = 0.4, nu: int = 256, nv: int = 128) -> trimesh.Trimesh:
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    U, V = np.meshgrid(u, v, indexing="ij")
    x = (R + r * np.cos(V)) * np.cos(U)
    y = (R + r * np.cos(V)) * np.sin(U)
    z = r * np.sin(V)
    verts = np.stack([x, y, z], -1).reshape(-1, 3)
    faces = []
    for i in range(nu):
        for j in range(nv):
            a = i * nv + j
            b = ((i + 1) % nu) * nv + j
            c = ((i + 1) % nu) * nv + (j + 1) % nv
            d = i * nv + (j + 1) % nv
            faces.append([a, b, c])
            faces.append([a, c, d])
    m = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    trimesh.repair.fix_normals(m)
    return m


def load_mesh(name: str) -> trimesh.Trimesh:
    if name == "sphere":
        return make_sphere()
    if name == "torus":
        return make_torus()
    path = STANFORD[name]
    m = trimesh.load(path, process=True)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    # keep the scan's own (consistent) winding; only merge dup verts via process=True
    return m


def bbox_diag(pts: np.ndarray) -> float:
    return float(np.linalg.norm(pts.max(0) - pts.min(0)))


def sample(mesh: trimesh.Trimesh, n: int, seed: int = 0):
    pts, fid = trimesh.sample.sample_surface(mesh, n, seed=seed)
    normals = mesh.face_normals[fid]
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    return np.asarray(pts, np.float64), np.asarray(normals, np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["sphere", "torus", "bunny", "armadillo", "dragon", "happy"])
    ap.add_argument("--n", type=int, nargs="+", default=[40000],
                    help="point counts to sample (first is the 'main' density)")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0025, 0.005, 0.01],
                    help="noise levels as fraction of bbox diagonal")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(MESH_DIR, exist_ok=True)
    os.makedirs(PC_DIR, exist_ok=True)

    for name in args.models:
        try:
            mesh = load_mesh(name)
        except Exception as e:
            print(f"[SKIP] {name}: {e!r}")
            continue
        mesh.export(f"{MESH_DIR}/{name}.ply")
        print(f"[{name}] mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
              f"watertight={mesh.is_watertight}")
        for n in args.n:
            pts, gtn = sample(mesh, n, seed=args.seed)
            diag = bbox_diag(pts)
            np.savez(f"{PC_DIR}/{name}_n{n}_clean.npz",
                     points=pts.astype(np.float32), gt_normals=gtn.astype(np.float32))
            rng = np.random.RandomState(args.seed + 1)
            for sig in args.sigmas:
                noise = rng.randn(*pts.shape) * (sig * diag)
                pc_noisy = pts + noise
                tag = f"{name}_n{n}_sigma{sig:g}".replace("0.", "p")
                np.savez(f"{PC_DIR}/{tag}.npz",
                         points=pc_noisy.astype(np.float32), gt_normals=gtn.astype(np.float32))
            print(f"  n={n}: clean + {len(args.sigmas)} noisy variants (diag={diag:.4f})")


if __name__ == "__main__":
    main()
