# Globally-Consistent Point-Cloud Normal Estimation — WNNC, a pure-PyTorch reimplementation, and a benchmark

《几何计算前沿》课程大作业 · 选题一（点云法向估计 + 泊松重建验证）

This project estimates **globally consistent (oriented) normals** for raw point
clouds and validates them by **screened Poisson surface reconstruction**. It
benchmarks three orientation methods on a common data/metric harness:

| method | what it is | source |
|---|---|---|
| `wnnc_cuda` | **WNNC** — Winding-Number Normal Consistency (ToG 2024), official CUDA treecode | baseline we build on (`jsnln/WNNC`) |
| `ours_torch` | **our from-scratch pure-PyTorch dense WNNC** (no custom CUDA) | `src/ours/wnnc_torch.py` |
| `pca_mst` | classic local PCA + MST orientation (Hoppe 1992) | Open3D |

Everything is scriptable and judged by **numbers** (oriented accuracy, angular
error, Chamfer, F-score, normal consistency) — no manual inspection.

---

## 1. Environment

Developed/tested on a single **RTX 4090 (24 GB, sm_89)** cloud instance:

- Ubuntu 22.04, **CUDA Toolkit 12.1** (`nvcc` 12.1), driver 570
- Python 3.12, **PyTorch 2.3.0+cu121**, managed with `uv`
- The only compiled component is the official WNNC CUDA extension.

### Install

```bash
# 0) python env (uv). torch must match the CUDA toolkit (cu121 for a 4090 image).
uv venv --python 3.12 .venv
uv pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install numpy scipy scikit-learn open3d trimesh pymeshlab matplotlib pandas tqdm plyfile

# 1) build the official WNNC CUDA treecode  (the single make-or-break step)
#    setting the 4090 arch avoids a "no kernel image" runtime error.
git clone https://github.com/jsnln/WNNC
cd WNNC/ext && TORCH_CUDA_ARCH_LIST=8.9 uv pip install -e . && cd ../..

# sanity: both must succeed
python -c "import torch; print(torch.cuda.get_device_capability())"     # (8, 9)
python -c "import wn_treecode; print('WNNC ext OK')"
```

> On this project's server we instead reused the image's pre-installed
> `torch 2.3.0+cu121` via `uv venv --system-site-packages .venv`, which avoids a
> 2.5 GB re-download; either route works. If GitHub is slow, enable an academic
> proxy (e.g. AutoDL's `source /etc/network_turbo`) *only* for the `git clone`.

---

## 2. Data

Ground-truth models are (a) synthetic **sphere/torus** (exact analytic normals,
watertight) and (b) four **Stanford 3D Scanning** scans (Bunny, Armadillo,
Dragon, Happy Buddha; GT normal = the source mesh's face normal). For each model
we draw surface samples and add Gaussian noise (σ ∈ {0.25, 0.5, 1}% of the bbox
diagonal). GT normals of a noisy point are the underlying clean-surface normals.

```bash
python scripts/download_data.py    # Stanford meshes -> data/meshes_raw/
python scripts/prepare_data.py     # sample points + normals -> data/pointclouds/, GT meshes -> data/meshes/
```

`data/pointclouds/<model>_n40000_<variant>.npz` holds `points (N,3)` and
`gt_normals (N,3)`; `variant ∈ {clean, sigmap0025, sigmap005, sigmap01}`.

---

## 3. Run

```bash
# one method on one cloud:
python scripts/run_normals.py data/pointclouds/dragon_n40000_clean.npz --method ours_torch --width_config l0
python scripts/eval_normals.py --pred results/normals/dragon_n40000_clean__ours_torch.npz \
                               --gt   data/pointclouds/dragon_n40000_clean.npz

# full benchmark  (5 models x 4 noise levels x 3 methods, + Poisson recon):
python scripts/benchmark.py --do_recon --poisson_depth 8
python scripts/make_figs.py                # tables + plots + renders -> results/csv, results/figs

# Innovation-A numerical alignment (ours vs official, same input):
python scripts/check_alignment.py --a results/normals/dragon_n40000_clean__ours_torch.npz \
                                  --b results/normals/dragon_n40000_clean__wnnc_cuda.npz
```

Outputs: `results/normals/*.npz` (oriented normals), `results/meshes/*.ply`
(Poisson meshes), `results/csv/*.csv` (all metrics + pivot tables),
`results/figs/*.png` (plots and renders).

---

## 4. Repository layout

```
src/ours/wnnc_torch.py     Innovation A: pure-PyTorch dense WNNC solver (the core new code)
scripts/
  download_data.py         fetch Stanford meshes
  prepare_data.py          sample GT point clouds + synthesise noise
  run_normals.py           unified runner for the 3 methods
  eval_normals.py          oriented accuracy / angular error
  poisson_recon.py         screened Poisson reconstruction (Open3D)
  eval_recon.py            Chamfer / F-score / normal consistency
  benchmark.py             orchestrate the full matrix -> CSVs
  make_figs.py             CSVs -> tables, plots, offscreen renders
  check_alignment.py       ours-vs-official numerical validation
WNNC/                      official repo (baseline; CUDA treecode lives in WNNC/ext)
report/report.md           the written report
results/                   produced artefacts (normals, meshes, csv, figs)
```

## 5. What is *ours* vs. baseline (see report §6 for the compliance argument)

The WNNC **algorithm/treecode** is the official `jsnln/WNNC` baseline. Our own
code is: the **pure-PyTorch dense WNNC** (`src/ours/wnnc_torch.py`), the unified
**3-method benchmark harness**, the **data + noise pipeline**, the **evaluation**
(normals + reconstruction) and the **Open3D Poisson** integration — i.e. this is
a benchmark/reimplementation study around WNNC, not a clone of it.
