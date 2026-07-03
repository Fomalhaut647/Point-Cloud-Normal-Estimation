"""Pure-PyTorch dense reimplementation of WNNC (Winding Number Normal Consistency).

This is *Innovation A* of the project: a from-scratch, dependency-free (no custom
CUDA extension) reimplementation of the WNNC normal-orientation solver of

    Lin, Shi, Liu. "Fast and Globally Consistent Normal Orientation based on the
    Winding Number Normal Consistency." ACM ToG 2024 (SIGGRAPH Asia 2024).

The official repository (``jsnln/WNNC``) accelerates the three core operators
A, A^T and G with a hand-written CUDA *treecode* (a Barnes--Hut style
approximation).  Here we instead evaluate the operators **densely** (exact,
all-pairs, O(N^2)) using batched tensor algebra, which:

  * needs *no* compiled CUDA extension (pure ``torch``), hence trivially portable;
  * is *more* accurate than the treecode (no far-field approximation), so it
    doubles as a numerical reference for validating the official kernels;
  * still runs on the GPU via chunked matrix algebra (a few seconds for 40k pts).

The pairwise kernels below are transcribed **verbatim** from the official CPU
kernels (``wn_treecode_cpu_kernels.cpp``); only the far-field approximation is
dropped.  With ``diff = x_i - y_j`` and ``d = |diff|`` and per-query smoothing
width ``w_i`` (contributions with ``d < w_i`` are hard-cut to zero):

    A(mu)_i  = sum_j [d>=w] * ( -(x_i - y_j) . mu_j / d^3 )                # -> scalar
    A^T(s)_i = sum_j [d>=w] * (  (x_i - y_j) * s_j     / d^3 )             # -> vector
    G(mu)_i  = sum_j [d>=w] * (  mu_j / d^3 - 3 (x_i-y_j)((x_i-y_j).mu_j)/d^5 )  # vector
"""

from __future__ import annotations

import argparse
import os
from time import time

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dense winding-number operators (exact all-pairs, chunked over the query axis)
# ---------------------------------------------------------------------------
class DenseWindingNumber:
    """Exact dense counterpart of ``wn_treecode.WindingNumberTreecode``.

    Parameters
    ----------
    points : (N, 3) tensor
        The source point set (also used as query points).
    block_pairs : int
        Target number of (query, source) pairs materialised at once.  The query
        chunk size is ``max(1, block_pairs // N)``.  Larger => faster but more
        memory.  ~1.2e8 keeps peak memory a few GB for N up to ~1e5 on a 24GB GPU.
    """

    def __init__(self, points: torch.Tensor, block_pairs: int = 120_000_000):
        assert points.dim() == 2 and points.shape[1] == 3
        self.points = points
        self.N = points.shape[0]
        self.device = points.device
        self.dtype = points.dtype
        self.block = max(1, int(block_pairs // max(1, self.N)))

    def _iter_query_blocks(self):
        for start in range(0, self.N, self.block):
            end = min(self.N, start + self.block)
            yield start, end

    @staticmethod
    def _safe_inv(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 1/x on kept entries, 0 elsewhere; avoids inf/nan from the d==0 self term.
        x = x.clamp_min(1e-24)
        return torch.where(mask, x.reciprocal(), torch.zeros_like(x))

    def forward_A(self, normals: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
        """(N,3) normals, (N,) widths -> (N,1) scalar field A(mu)."""
        P = self.points
        out = torch.empty(self.N, 1, device=self.device, dtype=self.dtype)
        for s, e in self._iter_query_blocks():
            diff = P[s:e, None, :] - P[None, :, :]          # [B,N,3] = x_i - y_j
            d2 = (diff * diff).sum(-1)                       # [B,N]
            d = d2.sqrt()
            mask = d >= widths[s:e, None]                    # [B,N] hard cutoff
            inv_d3 = self._safe_inv(d * d2, mask)            # [B,N]
            dot = (diff * normals[None, :, :]).sum(-1)       # (x_i-y_j).mu_j
            out[s:e, 0] = (-(dot) * inv_d3).sum(1)
            del diff, d2, d, mask, inv_d3, dot
        return out

    def forward_AT(self, values: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
        """(N,1) scalars, (N,) widths -> (N,3) vector field A^T(s)."""
        P = self.points
        s_j = values[:, 0]                                   # [N]
        out = torch.empty(self.N, 3, device=self.device, dtype=self.dtype)
        for s, e in self._iter_query_blocks():
            diff = P[s:e, None, :] - P[None, :, :]           # [B,N,3]
            d2 = (diff * diff).sum(-1)
            d = d2.sqrt()
            mask = d >= widths[s:e, None]
            coef = self._safe_inv(d * d2, mask) * s_j[None, :]   # [B,N]
            out[s:e] = (diff * coef[..., None]).sum(1)
            del diff, d2, d, mask, coef
        return out

    def forward_G(self, normals: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
        """(N,3) normals, (N,) widths -> (N,3) WNNC gradient field G(mu)."""
        P = self.points
        out = torch.empty(self.N, 3, device=self.device, dtype=self.dtype)
        for s, e in self._iter_query_blocks():
            diff = P[s:e, None, :] - P[None, :, :]           # [B,N,3]
            d2 = (diff * diff).sum(-1)
            d = d2.sqrt()
            mask = d >= widths[s:e, None]
            inv_d3 = self._safe_inv(d * d2, mask)            # [B,N]
            inv_d5 = self._safe_inv(d2 * d * d2, mask)       # [B,N] = 1/d^5
            dot = (diff * normals[None, :, :]).sum(-1)       # [B,N]
            term = normals[None, :, :] * inv_d3[..., None] \
                - 3.0 * diff * (dot * inv_d5)[..., None]
            out[s:e] = term.sum(1)
            del diff, d2, d, mask, inv_d3, inv_d5, dot, term
        return out


PRESET_WIDTHS = {
    "l0": [0.002, 0.016],   # clean / uniform / noise-free
    "l1": [0.01, 0.04],     # real scans / small noise
    "l2": [0.02, 0.08],     # sigma=0.25%
    "l3": [0.03, 0.12],     # sigma=0.5%
    "l4": [0.04, 0.16],     # sigma=1%
    "l5": [0.05, 0.20],     # sparse / sketches
}


def normalize_points(points_unnormalized: np.ndarray, bbox_scale: float = 1.1):
    """Same normalisation as the official ``main_wnnc.py``."""
    center = (points_unnormalized.min(0) + points_unnormalized.max(0)) / 2.0
    bbox_len = (points_unnormalized.max(0) - points_unnormalized.min(0)).max()
    pts = (points_unnormalized - center) * (2.0 / (bbox_len * bbox_scale))
    return pts, center, bbox_len


def solve_normals(
    points_unnormalized: np.ndarray,
    width_config: "str | None" = "l0",
    wsmin: float | None = None,
    wsmax: float | None = None,
    iters: int = 40,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    block_pairs: int = 120_000_000,
    verbose: bool = False,
):
    """Run the dense WNNC solver; returns (out_normals[N,3], elapsed_seconds).

    The iteration is a line-by-line port of the official ``main_wnnc.py`` loop
    (a conjugate-gradient-flavoured steepest step on ``A^T A mu = A^T b`` with a
    per-iteration WNNC re-projection through G and an annealed smoothing width).
    """
    if wsmin is None or wsmax is None:
        wsmin, wsmax = PRESET_WIDTHS[width_config]
    assert wsmin <= wsmax

    pts_np, _, _ = normalize_points(points_unnormalized)
    points = torch.from_numpy(pts_np).contiguous().to(dtype).to(device)
    normals = torch.zeros_like(points)
    b = torch.ones(points.shape[0], 1, dtype=dtype, device=device) * 0.5
    widths = torch.ones_like(points[:, 0])

    wn = DenseWindingNumber(points, block_pairs=block_pairs)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time()
    with torch.no_grad():
        out_normals = normals
        for i in range(iters):
            width_scale = wsmin + ((iters - 1 - i) / (iters - 1)) * (wsmax - wsmin)
            w = widths * width_scale

            A_mu = wn.forward_A(normals, w)
            AT_A_mu = wn.forward_AT(A_mu, w)
            r = wn.forward_AT(b, w) - AT_A_mu
            A_r = wn.forward_A(r, w)
            alpha = (r * r).sum() / (A_r * A_r).sum()
            normals = normals + alpha * r

            out_normals = wn.forward_G(normals, w)
            out_normals = F.normalize(out_normals, dim=-1).contiguous()
            normals_len = torch.linalg.norm(normals, dim=-1, keepdim=True)
            normals = out_normals.clone() * normals_len
            if verbose:
                print(f"  [ours] iter {i:02d} width={width_scale:.4f} alpha={alpha.item():.3e}")
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time() - t0
    return out_normals.detach().cpu().numpy(), elapsed


def _load_points(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".xyz":
        return np.loadtxt(path)[:, :3]
    if ext == ".npy":
        return np.load(path)[:, :3]
    if ext == ".npz":
        return np.load(path)["points"][:, :3]
    if ext in (".ply", ".obj"):
        import trimesh
        return np.asarray(trimesh.load(path, process=False).vertices)
    raise ValueError(f"unsupported extension: {ext}")


def main():
    ap = argparse.ArgumentParser(description="Pure-PyTorch dense WNNC normal estimation")
    ap.add_argument("input")
    ap.add_argument("--width_config", default="l0", choices=list(PRESET_WIDTHS) + ["custom"])
    ap.add_argument("--wsmin", type=float, default=None)
    ap.add_argument("--wsmax", type=float, default=None)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--dtype", default="float", choices=["float", "double"])
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--block_pairs", type=int, default=120_000_000)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    pts = _load_points(args.input)
    dtype = torch.float32 if args.dtype == "float" else torch.float64
    device = "cpu" if args.cpu else "cuda"
    wc = "custom" if args.width_config == "custom" else args.width_config
    wsmin, wsmax = (args.wsmin, args.wsmax) if wc == "custom" else (None, None)

    normals, elapsed = solve_normals(
        pts, width_config=(None if wc == "custom" else wc),
        wsmin=wsmin, wsmax=wsmax, iters=args.iters,
        dtype=dtype, device=device, block_pairs=args.block_pairs, verbose=args.verbose,
    )
    print(f"[ours] time_main: {elapsed:.4f} s  ({len(pts)} points)")
    os.makedirs(args.out_dir, exist_ok=True)
    out = np.concatenate([pts, normals], -1)
    dst = os.path.join(args.out_dir, os.path.basename(args.input).rsplit(".", 1)[0] + ".xyz")
    np.savetxt(dst, out)
    print(f"[ours] saved -> {dst}")


if __name__ == "__main__":
    main()
