from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REQUESTED_FILES = [
    "faithfulness_correlation_ampliada_distribucion.png",
    "faithfulness_correlation_ampliada_por_metodo.png",
    "faithfulness_occlusion_ampliada_por_resultado.png",
    "kepler_error_detection_auc_from_xai_metrics.png",
    "kepler_xai_explanation_map_high_confidence_true_positive_idx1132.png",
    "tess_error_detection_auc_from_xai_metrics.png",
    "tess_xai_explanation_map_false_negative_idx495.png",
]

METHOD_LABELS = {
    "saliency": "Saliency",
    "smoothgrad": "SmoothGrad",
    "integrated_gradients": "Integrated Gradients",
    "occlusion": "Occlusion",
    "gradcam": "Grad-CAM 1D",
    "gradcam_multiscale": "Multi-scale Grad-CAM",
    "consensus": "Consensus",
}

OUTCOME_LABELS = {
    "TN": "True negative",
    "FP": "False positive",
    "FN": "False negative",
    "TP": "True positive",
}

CASE_LABELS = {
    "positivo_correcto_alta_confianza": "high-confidence true positive",
    "negativo_correcto_alta_confianza": "high-confidence true negative",
    "cerca_del_umbral": "near decision threshold",
    "falso_positivo": "false positive",
    "falso_negativo": "false negative",
    "mayor_incertidumbre_mc_dropout": "highest MC Dropout uncertainty",
    "high_confidence_true_positive": "high-confidence true positive",
    "false_negative": "false negative",
}

METRIC_LABELS = {
    "total_focus_count": "Attention focus count",
    "mean_attention_position_variance": "Attention position variance",
    "mean_attention_entropy": "Attention entropy",
    "mean_central_importance_ratio": "Central importance ratio",
    "attention_quality_score": "Attention quality score",
    "attention_risk_score": "Attention risk score",
    "mc_std": "MC Dropout std.",
    "target_confidence": "Target confidence",
    "maxlike_entropy": "MaxLike entropy",
    "mc_dropout_std_probability": "MC Dropout probability std.",
    "aleatoric_uncertainty": "Aleatoric uncertainty",
    "epistemic_uncertainty": "Epistemic uncertainty",
}


def find_repo_root(start: Path) -> Path:
    for parent in [start.resolve(), *start.resolve().parents]:
        if (parent / "xai_mejores_modelos.py").exists() and (parent / "notebook").exists():
            return parent
    raise FileNotFoundError("Could not find the entrega2/github repository root.")


def import_xai_module(repo_root: Path):
    module_path = repo_root / "xai_mejores_modelos.py"
    spec = importlib.util.spec_from_file_location("xai_mejores_modelos_local", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def plot_faithfulness_distribution(repo_root: Path, out_dir: Path) -> None:
    sample_path = repo_root / "notebook" / "resultados" / "figuras_academicas" / "fidelidad_ampliada" / "faithfulness_ampliada_by_sample.csv"
    df = pd.read_csv(sample_path)
    df = df[df["valid_correlation"].astype(bool)].copy()
    df["method_label"] = df["method"].map(METHOD_LABELS).fillna(df["method"])
    df["dataset_label"] = df["dataset"].str.upper()

    datasets = [d for d in ["KEPLER", "TESS"] if d in set(df["dataset_label"])]
    methods = ["Saliency", "Integrated Gradients", "Occlusion", "Consensus"]
    colors = {"KEPLER": "#3d5a80", "TESS": "#e76f51"}

    fig, axes = plt.subplots(1, 2, figsize=(15.4, 5.7), constrained_layout=True)
    for ax, corr_col, title in [
        (axes[0], "faithfulness_pearson", "Pearson Faithfulness Correlation"),
        (axes[1], "faithfulness_spearman", "Spearman Faithfulness Correlation"),
    ]:
        positions = []
        data = []
        labels = []
        pos = 1.0
        for method in methods:
            for dataset in datasets:
                vals = df.loc[(df["method_label"] == method) & (df["dataset_label"] == dataset), corr_col].dropna().to_numpy()
                if vals.size:
                    data.append(vals)
                    positions.append(pos)
                    labels.append(f"{method}\n{dataset}")
                pos += 0.82
            pos += 0.38
        parts = ax.violinplot(data, positions=positions, widths=0.62, showmeans=False, showmedians=True)
        for body, label in zip(parts["bodies"], labels):
            dataset = label.split("\n")[-1]
            body.set_facecolor(colors.get(dataset, "#6c757d"))
            body.set_edgecolor("#263238")
            body.set_alpha(0.68)
        for key in ["cbars", "cmins", "cmaxes", "cmedians"]:
            if key in parts:
                parts[key].set_color("#263238")
                parts[key].set_linewidth(0.8)
        ax.axhline(0.0, color="#202020", linestyle="--", linewidth=0.9, alpha=0.75)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Correlation")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Expanded faithfulness analysis - correlation distributions", fontsize=14, fontweight="bold")
    fig.savefig(out_dir / "faithfulness_correlation_ampliada_distribucion.png", dpi=220)
    plt.close(fig)


def plot_faithfulness_by_method(repo_root: Path, out_dir: Path) -> None:
    summary_path = repo_root / "notebook" / "resultados" / "figuras_academicas" / "fidelidad_ampliada" / "faithfulness_ampliada_summary.csv"
    df = pd.read_csv(summary_path)
    df["method_label"] = df["method"].map(METHOD_LABELS).fillna(df["method"])
    df["dataset_label"] = df["dataset"].str.upper()

    datasets = [d for d in ["kepler", "tess"] if d in set(df["dataset"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.0 * len(datasets), 5.5), constrained_layout=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        sub = df[df["dataset"] == dataset].sort_values("pearson_mean")
        y = np.arange(len(sub))
        ax.barh(y, sub["pearson_mean"], xerr=sub["pearson_std"].fillna(0.0), color="#457b9d", alpha=0.86, capsize=3)
        ax.axvline(0.0, color="#202020", linestyle="--", linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["method_label"])
        ax.set_xlabel("Mean Pearson Faithfulness Correlation")
        ax.set_title(f"{dataset.upper()} - faithfulness by XAI method", fontweight="bold")
        ax.grid(axis="x", alpha=0.22)
        for yi, value, n in zip(y, sub["pearson_mean"], sub["valid_n"]):
            ax.text(float(value) + (0.01 if value >= 0 else -0.01), yi, f"n={int(n)}", va="center", ha="left" if value >= 0 else "right", fontsize=8.5)
    fig.suptitle("Expanded faithfulness analysis by method", fontsize=14, fontweight="bold")
    fig.savefig(out_dir / "faithfulness_correlation_ampliada_por_metodo.png", dpi=220)
    plt.close(fig)


def plot_occlusion_by_outcome(repo_root: Path, out_dir: Path) -> None:
    outcome_path = repo_root / "notebook" / "resultados" / "figuras_academicas" / "fidelidad_ampliada" / "faithfulness_ampliada_by_outcome.csv"
    df = pd.read_csv(outcome_path)
    df = df[df["method"] == "occlusion"].copy()
    df["outcome_label"] = df["outcome"].map(OUTCOME_LABELS).fillna(df["outcome"])

    order = ["TN", "FP", "FN", "TP"]
    colors = {"TN": "#3d5a80", "FP": "#e07a5f", "FN": "#c1121f", "TP": "#2a9d8f"}
    datasets = [d for d in ["kepler", "tess"] if d in set(df["dataset"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.0 * len(datasets), 5.4), constrained_layout=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        sub = df[df["dataset"] == dataset].set_index("outcome").reindex(order).dropna(subset=["pearson_mean"])
        y = np.arange(len(sub))
        ax.barh(y, sub["pearson_mean"], color=[colors.get(o, "#6c757d") for o in sub.index], alpha=0.86)
        ax.axvline(0.0, color="#202020", linestyle="--", linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([OUTCOME_LABELS.get(o, o) for o in sub.index])
        ax.set_xlabel("Mean Pearson Faithfulness Correlation")
        ax.set_title(f"{dataset.upper()} - Occlusion faithfulness by outcome", fontweight="bold")
        ax.grid(axis="x", alpha=0.22)
        for yi, value, n in zip(y, sub["pearson_mean"], sub["n"]):
            ax.text(float(value) + (0.01 if value >= 0 else -0.01), yi, f"n={int(n)}", va="center", ha="left" if value >= 0 else "right", fontsize=8.5)
    fig.suptitle("Expanded occlusion faithfulness by prediction outcome", fontsize=14, fontweight="bold")
    fig.savefig(out_dir / "faithfulness_occlusion_ampliada_por_resultado.png", dpi=220)
    plt.close(fig)


def plot_error_detection(repo_root: Path, out_dir: Path) -> None:
    metrics_path = repo_root / "notebook" / "resultados" / "tablas" / "xai_error_detection_metrics.csv"
    df = pd.read_csv(metrics_path)
    for dataset in ["kepler", "tess"]:
        sub = df[df["dataset"] == dataset].sort_values("roc_auc_best_direction", ascending=False).head(10)
        sub = sub.sort_values("roc_auc_best_direction")
        labels = sub["metric"].map(METRIC_LABELS).fillna(sub["metric"])
        fig, ax = plt.subplots(figsize=(11.0, 6.3), constrained_layout=True)
        ax.barh(labels, sub["roc_auc_best_direction"], color="#e76f51", alpha=0.86)
        ax.axvline(0.5, color="#202020", linestyle="--", linewidth=1.0)
        ax.set_xlim(0.45, 1.0)
        ax.set_xlabel("ROC-AUC for separating errors from correct predictions")
        ax.set_title(f"{dataset.upper()} - XAI-metric potential for error detection", fontweight="bold", fontsize=13)
        ax.grid(axis="x", alpha=0.22)
        for value, y in zip(sub["roc_auc_best_direction"], labels):
            ax.text(float(value) + 0.006, y, f"{float(value):.3f}", va="center", fontsize=8.8)
        fig.savefig(out_dir / f"{dataset}_error_detection_auc_from_xai_metrics.png", dpi=220)
        plt.close(fig)


def plot_heatmap_line_en(ax: plt.Axes, xai, curve: np.ndarray, heat: np.ndarray, title: str) -> None:
    x = np.arange(len(curve))
    start, end = xai.transit_window(len(curve))
    ymin = float(np.nanmin(curve))
    ymax = float(np.nanmax(curve))
    if abs(ymax - ymin) < 1e-6:
        ymin -= 1.0
        ymax += 1.0
    pad = 0.06 * (ymax - ymin)
    ax.imshow(
        normalize_01(heat)[None, :],
        aspect="auto",
        cmap="magma",
        extent=[0, len(curve) - 1, ymin - pad, ymax + pad],
        origin="lower",
        alpha=0.42,
    )
    ax.axvspan(start, end - 1, color="#2f9e44", alpha=0.12, lw=0)
    ax.axvline(len(curve) // 2, color="#2f9e44", linestyle="--", linewidth=0.85, alpha=0.65)
    ax.plot(x, curve, color="#101010", lw=0.85)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("flux")
    ax.set_xlim(0, len(curve) - 1)
    ax.grid(alpha=0.18, linewidth=0.6)


def plot_explanation_en(
    xai,
    dataset: str,
    case_label: str,
    idx: int,
    xg: np.ndarray,
    xl: np.ndarray,
    maps: dict[str, tuple[np.ndarray, np.ndarray]],
    y_true: int,
    prob: float,
    threshold: float,
    mc_mean: float,
    mc_std: float,
    out_path: Path,
) -> None:
    panels = [
        ("saliency", "Saliency Map"),
        ("smoothgrad", "SmoothGrad"),
        ("integrated_gradients", "Integrated Gradients"),
        ("occlusion", "Occlusion Sensitivity"),
        ("gradcam", "Grad-CAM 1D (diagnostic)"),
        ("gradcam_multiscale", "Multi-scale Grad-CAM"),
        ("consensus", "Input consensus"),
    ]
    fig, axes = plt.subplots(len(panels), 2, figsize=(15, 19), constrained_layout=True)
    fig.suptitle(
        f"{dataset.upper()} | {case_label} | idx={idx} | y={y_true} | "
        f"p={prob:.3f} | thr={threshold:.2f} | MC={mc_mean:.3f}+/-{mc_std:.3f}",
        fontsize=12,
        fontweight="bold",
    )
    for row, (key, title) in enumerate(panels):
        g, l = maps[key]
        plot_heatmap_line_en(axes[row, 0], xai, xg, g, f"Global view - {title}")
        plot_heatmap_line_en(axes[row, 1], xai, xl, l, f"Local view - {title}")
    axes[-1, 0].set_xlabel("time index")
    axes[-1, 1].set_xlabel("time index")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def generate_xai_map(
    repo_root: Path,
    data_root: Path,
    model_root: Path,
    out_dir: Path,
    dataset: str,
    idx: int,
    case_key: str,
    out_name: str,
    device_name: str,
    smooth_samples: int,
    ig_steps: int,
) -> None:
    xai = import_xai_module(repo_root)
    device = torch.device(device_name)
    spec = xai.DATASETS[dataset]
    model = xai.build_model(model_root, spec, device)
    threshold = xai.load_threshold(model_root, spec)
    xg, xl, y, _raw = xai.load_dataset(data_root, spec)
    if idx >= len(y):
        raise IndexError(f"Index {idx} is outside {dataset} test set with {len(y)} samples.")

    with torch.no_grad():
        logits = model(
            torch.from_numpy(xg[idx : idx + 1]).unsqueeze(1).to(device),
            torch.from_numpy(xl[idx : idx + 1]).unsqueeze(1).to(device),
        )
        prob = float(torch.sigmoid(logits)[0, 0].detach().cpu().item())
    pred_label = int(prob >= threshold)

    mc_path = repo_root / "notebook" / "resultados" / "tablas" / "mc_dropout_summary.csv"
    mc_df = pd.read_csv(mc_path)
    mc_row = mc_df[(mc_df["dataset"] == dataset) & (mc_df["index"] == idx)]
    if len(mc_row):
        mc_mean = float(mc_row.iloc[0]["mc_mean"])
        mc_std = float(mc_row.iloc[0]["mc_std"])
    else:
        mc_mean = prob
        mc_std = float("nan")

    sal_g, sal_l = xai.compute_saliency(model, xg[idx], xl[idx], device, target_label=pred_label)
    smooth_g, smooth_l = xai.compute_smoothgrad(model, xg[idx], xl[idx], device, target_label=pred_label, samples=smooth_samples)
    ig_g, ig_l = xai.compute_integrated_gradients(model, xg[idx], xl[idx], device, target_label=pred_label, steps=ig_steps)
    cam_g, cam_l = xai.compute_gradcam(model, xg[idx], xl[idx], device, target_label=pred_label)
    mcam_g, mcam_l = xai.compute_multiscale_gradcam(model, xg[idx], xl[idx], device, target_label=pred_label)
    occ_g, occ_l = xai.compute_occlusion_sensitivity(model, xg[idx], xl[idx], device, target_label=pred_label)
    cons_g = xai.consensus_importance(smooth_g, ig_g, occ_g, mcam_g)
    cons_l = xai.consensus_importance(smooth_l, ig_l, occ_l, mcam_l)

    maps = {
        "saliency": (sal_g, sal_l),
        "smoothgrad": (smooth_g, smooth_l),
        "integrated_gradients": (ig_g, ig_l),
        "occlusion": (occ_g, occ_l),
        "gradcam": (cam_g, cam_l),
        "gradcam_multiscale": (mcam_g, mcam_l),
        "consensus": (cons_g, cons_l),
    }
    plot_explanation_en(
        xai=xai,
        dataset=dataset,
        case_label=CASE_LABELS[case_key],
        idx=idx,
        xg=xg[idx],
        xl=xl[idx],
        maps=maps,
        y_true=int(y[idx]),
        prob=prob,
        threshold=threshold,
        mc_mean=mc_mean,
        mc_std=mc_std,
        out_path=out_dir / out_name,
    )


def write_readme(out_dir: Path) -> None:
    text = """# Missing English Figures for Overleaf

This folder contains English-language replacements for the seven figures requested for the Overleaf manuscript. The filenames are intentionally preserved exactly, so the files can be copied over the previous versions without changing LaTeX paths.

All visible titles, axes, panel labels and captions inside the PNG files are in English. The faithfulness and error-detection figures are regenerated from the CSV tables. The two XAI maps are recomputed from the saved champion models and test samples, explaining the model-predicted class.

## Files

```text
faithfulness_correlation_ampliada_distribucion.png
faithfulness_correlation_ampliada_por_metodo.png
faithfulness_occlusion_ampliada_por_resultado.png
kepler_error_detection_auc_from_xai_metrics.png
kepler_xai_explanation_map_high_confidence_true_positive_idx1132.png
tess_error_detection_auc_from_xai_metrics.png
tess_xai_explanation_map_false_negative_idx495.png
```

## Reproducibility

From the repository root, run:

```bash
python notebook/resultados/figuras_academicas/imagenes_que_faltan_en_ingles/generar_imagenes_ingles.py
```

If the Kepler HDF5 file is stored outside the repository, pass `--data-root /path/to/project_with_data`.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    default_repo = find_repo_root(Path(__file__).resolve())
    parser = argparse.ArgumentParser(description="Regenerate the seven requested Overleaf figures with English text.")
    parser.add_argument("--repo-root", type=Path, default=default_repo, help="Repository root containing notebook/, xai_mejores_modelos.py and result tables.")
    parser.add_argument("--data-root", type=Path, default=None, help="Root containing datos_procesados_h5/ and tfrecords_TESS/. Defaults to repo-root.")
    parser.add_argument("--model-root", type=Path, default=None, help="Root containing mejores_resultados/modelos/. Defaults to repo-root.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smooth-samples", type=int, default=24)
    parser.add_argument("--ig-steps", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    data_root = (args.data_root or repo_root).resolve()
    model_root = (args.model_root or repo_root).resolve()
    out_dir = repo_root / "notebook" / "resultados" / "figuras_academicas" / "imagenes_que_faltan_en_ingles"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9,
        "figure.titlesize": 14,
    })

    plot_faithfulness_distribution(repo_root, out_dir)
    plot_faithfulness_by_method(repo_root, out_dir)
    plot_occlusion_by_outcome(repo_root, out_dir)
    plot_error_detection(repo_root, out_dir)
    generate_xai_map(
        repo_root=repo_root,
        data_root=data_root,
        model_root=model_root,
        out_dir=out_dir,
        dataset="kepler",
        idx=1132,
        case_key="high_confidence_true_positive",
        out_name="kepler_xai_explanation_map_high_confidence_true_positive_idx1132.png",
        device_name=args.device,
        smooth_samples=args.smooth_samples,
        ig_steps=args.ig_steps,
    )
    generate_xai_map(
        repo_root=repo_root,
        data_root=data_root,
        model_root=model_root,
        out_dir=out_dir,
        dataset="tess",
        idx=495,
        case_key="false_negative",
        out_name="tess_xai_explanation_map_false_negative_idx495.png",
        device_name=args.device,
        smooth_samples=args.smooth_samples,
        ig_steps=args.ig_steps,
    )
    write_readme(out_dir)

    missing = [name for name in REQUESTED_FILES if not (out_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing requested files: {missing}")
    print(f"Generated {len(REQUESTED_FILES)} English figures in {out_dir}")


if __name__ == "__main__":
    main()
