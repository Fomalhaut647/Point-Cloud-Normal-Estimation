"""Turn the benchmark CSVs into report-ready tables and figures (all offscreen).

Outputs
-------
results/csv/table_normals_oriented_acc.csv / .md   pivot: method x (model,variant)
results/csv/table_normals_ang_err.csv / .md
results/csv/table_recon_chamfer.csv / .md
results/figs/acc_vs_noise.png        oriented accuracy vs noise level (per method)
results/figs/angerr_vs_noise.png     mean angular error vs noise level
results/figs/chamfer_vs_noise.png    reconstruction Chamfer vs noise level
results/figs/runtime.png             wall-clock per method (log scale)
results/figs/normals_<case>.png      normals-as-RGB point renders (method comparison)
results/figs/orient_<case>.png       orientation-correctness renders (green=ok, red=flipped)
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = "results/figs"
CSV = "results/csv"
NOISE_ORDER = ["clean", "sigmap0025", "sigmap005", "sigmap01"]
NOISE_X = {"clean": 0.0, "sigmap0025": 0.25, "sigmap005": 0.5, "sigmap01": 1.0}
METHOD_LABEL = {"wnnc_cuda": "WNNC (official CUDA)",
                "ours_torch": "Ours (pure PyTorch)",
                "pca_mst": "PCA+MST (Hoppe'92)"}
METHOD_COLOR = {"wnnc_cuda": "#1f77b4", "ours_torch": "#ff7f0e", "pca_mst": "#2ca02c"}


def _save_table(df, path_stem):
    df.to_csv(f"{path_stem}.csv")
    with open(f"{path_stem}.md", "w") as f:
        f.write(df.to_markdown(floatfmt=".3f"))


def summary_tables(ndf, rdf):
    for metric, stem in [("oriented_acc", "table_normals_oriented_acc"),
                         ("mean_ang_err_oriented", "table_normals_ang_err")]:
        piv = ndf.pivot_table(index="method", columns=["model", "variant"], values=metric)
        _save_table(piv, f"{CSV}/{stem}")
    # aggregate over models: mean per (method, variant)
    agg = ndf.groupby(["method", "variant"]).agg(
        oriented_acc=("oriented_acc", "mean"),
        mean_ang_err=("mean_ang_err_oriented", "mean"),
        time_sec=("time_sec", "mean")).reset_index()
    agg.to_csv(f"{CSV}/summary_by_noise.csv", index=False)
    if rdf is not None and len(rdf):
        piv = rdf.pivot_table(index="method", columns=["model", "variant"],
                              values="chamfer_l1_x1e3")
        _save_table(piv, f"{CSV}/table_recon_chamfer")
    return agg


def report_tables(ndf, rdf):
    """Compact, report-ready tables: rows = model x variant, cols = 3 methods."""
    models = ["sphere", "torus", "bunny", "armadillo", "dragon"]
    variants = ["clean", "sigmap0025", "sigmap005", "sigmap01"]
    idx = pd.MultiIndex.from_product([models, variants], names=["model", "variant"])
    methods = ["wnnc_cuda", "ours_torch", "pca_mst"]

    def emit(df, value, stem, fmt):
        piv = df.pivot_table(index=["model", "variant"], columns="method", values=value)
        piv = piv.reindex(idx)[methods]
        piv.columns = [METHOD_LABEL[m] for m in methods]
        piv.to_csv(f"{CSV}/{stem}.csv")
        with open(f"{CSV}/{stem}.md", "w") as f:
            f.write(piv.to_markdown(floatfmt=fmt))

    emit(ndf, "oriented_acc", "report_oriented_acc", ".3f")
    emit(ndf, "mean_ang_err_oriented", "report_ang_err_oriented", ".1f")
    emit(ndf, "mean_ang_err_unsigned", "report_ang_err_unsigned", ".1f")
    if rdf is not None and len(rdf):
        emit(rdf, "chamfer_l1_x1e3", "report_chamfer_l1", ".2f")
        emit(rdf, "normal_consistency", "report_normal_consistency", ".3f")


def line_vs_noise(ndf, metric, ylabel, out, logy=False):
    plt.figure(figsize=(6, 4))
    for method in ["wnnc_cuda", "ours_torch", "pca_mst"]:
        sub = ndf[ndf.method == method]
        xs, ys, es = [], [], []
        for v in NOISE_ORDER:
            vals = sub[sub.variant == v][metric].values
            if len(vals):
                xs.append(NOISE_X[v]); ys.append(np.mean(vals)); es.append(np.std(vals))
        if xs:
            plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                         label=METHOD_LABEL[method], color=METHOD_COLOR[method])
    plt.xlabel("Gaussian noise  (% of bbox diagonal)")
    plt.ylabel(ylabel)
    if logy:
        plt.yscale("log")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out, dpi=140); plt.close()


def chamfer_vs_noise(rdf, out):
    if rdf is None or not len(rdf):
        return
    plt.figure(figsize=(6, 4))
    for method in ["wnnc_cuda", "ours_torch", "pca_mst"]:
        sub = rdf[rdf.method == method]
        xs, ys = [], []
        for v in NOISE_ORDER:
            vals = sub[sub.variant == v]["chamfer_l1_x1e3"].values
            if len(vals):
                xs.append(NOISE_X[v]); ys.append(np.mean(vals))
        if xs:
            plt.plot(xs, ys, marker="s", label=METHOD_LABEL[method], color=METHOD_COLOR[method])
    plt.xlabel("Gaussian noise  (% of bbox diagonal)")
    plt.ylabel("Chamfer-L1 to GT  (x1e-3 of diag)")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out, dpi=140); plt.close()


def runtime_bar(ndf, out):
    plt.figure(figsize=(5, 4))
    order = ["pca_mst", "wnnc_cuda", "ours_torch"]
    means = [ndf[ndf.method == m]["time_sec"].mean() for m in order]
    colors = [METHOD_COLOR[m] for m in order]
    bars = plt.bar([METHOD_LABEL[m] for m in order], means, color=colors)
    plt.yscale("log")
    plt.ylabel("mean wall-clock per cloud (s, log)")
    for b, v in zip(bars, means):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}s", ha="center", va="bottom")
    plt.xticks(rotation=15, ha="right"); plt.tight_layout()
    plt.savefig(out, dpi=140); plt.close()


def _view(ax, pts):
    ax.set_box_aspect((1, 1, 1))
    c = pts.mean(0)
    r = np.percentile(np.linalg.norm(pts - c, axis=1), 99.5)  # robust to noise outliers
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=30)
    ax.margins(0)


def render_case(model, variant, methods, n=8000, seed=0):
    """Two rows of renders: normals-as-RGB, and orientation-correctness."""
    gt = np.load(f"data/pointclouds/{model}_n40000_{variant}.npz")
    gtn = gt["gt_normals"]
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(gt["points"]), min(n, len(gt["points"])), replace=False)

    fig1, axes1 = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4),
                               subplot_kw={"projection": "3d"})
    fig2, axes2 = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4),
                               subplot_kw={"projection": "3d"})
    if len(methods) == 1:
        axes1, axes2 = [axes1], [axes2]
    for ax1, ax2, method in zip(axes1, axes2, methods):
        f = f"results/normals/{model}_n40000_{variant}__{method}.npz"
        if not os.path.exists(f):
            continue
        d = np.load(f)
        pts, nrm = d["points"][idx], d["normals"][idx]
        rgb = np.clip(nrm * 0.5 + 0.5, 0, 1)
        ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=rgb, s=2)
        ax1.set_title(METHOD_LABEL[method], fontsize=10); _view(ax1, pts)
        ok = (nrm * gtn[idx]).sum(1) > 0
        col = np.where(ok[:, None], np.array([[0.1, 0.7, 0.1]]), np.array([[0.85, 0.1, 0.1]]))
        ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=col, s=2)
        ax2.set_title(f"{METHOD_LABEL[method]}\noriented {ok.mean()*100:.1f}%", fontsize=10)
        _view(ax2, pts)
    fig1.suptitle(f"{model} / {variant}: estimated normals (RGB = normal direction)")
    fig2.suptitle(f"{model} / {variant}: orientation correctness (green=correct, red=flipped)")
    fig1.tight_layout(); fig2.tight_layout()
    fig1.savefig(f"{FIG}/normals_{model}_{variant}.png", dpi=130)
    fig2.savefig(f"{FIG}/orient_{model}_{variant}.png", dpi=130)
    plt.close(fig1); plt.close(fig2)


def ablation_density_fig(out_acc, out_time):
    path = f"{CSV}/ablation_density.csv"
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    for metric, ylabel, out, logy in [
            ("oriented_acc", "Oriented accuracy", out_acc, False),
            ("time_sec", "wall-clock (s)", out_time, True)]:
        plt.figure(figsize=(6, 4))
        for method in ["wnnc_cuda", "ours_torch", "pca_mst"]:
            sub = df[df.method == method].groupby("n_points_req")[metric].mean()
            if len(sub):
                plt.plot(sub.index, sub.values, marker="o",
                         label=METHOD_LABEL[method], color=METHOD_COLOR[method])
        plt.xscale("log")
        if logy:
            plt.yscale("log")
        plt.xlabel("point count N (log)"); plt.ylabel(ylabel)
        plt.grid(alpha=0.3, which="both"); plt.legend(); plt.tight_layout()
        plt.savefig(out, dpi=140); plt.close()


def ablation_depth_fig(out):
    path = f"{CSV}/ablation_depth.csv"
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(df.depth, df["chamfer_l1_x1e3"], "o-", color="#1f77b4", label="Chamfer-L1")
    ax1.set_xlabel("Poisson octree depth"); ax1.set_ylabel("Chamfer-L1 (x1e-3)", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(df.depth, df["fscore@0.01"], "s--", color="#d62728", label="F-score@1%")
    ax2.set_ylabel("F-score@1%", color="#d62728")
    ax1.set_xticks(df.depth.tolist()); ax1.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    ndf = pd.read_csv(f"{CSV}/normals.csv")
    rdf = pd.read_csv(f"{CSV}/recon.csv") if os.path.exists(f"{CSV}/recon.csv") else None

    agg = summary_tables(ndf, rdf)
    report_tables(ndf, rdf)
    print(agg.to_string(index=False))
    line_vs_noise(ndf, "oriented_acc", "Oriented accuracy", f"{FIG}/acc_vs_noise.png")
    line_vs_noise(ndf, "mean_ang_err_oriented", "Mean angular error (deg)",
                  f"{FIG}/angerr_vs_noise.png", logy=True)
    chamfer_vs_noise(rdf, f"{FIG}/chamfer_vs_noise.png")
    runtime_bar(ndf, f"{FIG}/runtime.png")
    ablation_density_fig(f"{FIG}/ablation_density_acc.png", f"{FIG}/ablation_density_time.png")
    ablation_depth_fig(f"{FIG}/ablation_depth.png")

    methods = ["wnnc_cuda", "ours_torch", "pca_mst"]
    for model, variant in [("dragon", "clean"), ("dragon", "sigmap01"),
                           ("armadillo", "sigmap0025"), ("torus", "sigmap01")]:
        try:
            render_case(model, variant, methods)
            print(f"rendered {model}/{variant}")
        except Exception as e:
            print(f"render {model}/{variant} failed: {e!r}")
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
