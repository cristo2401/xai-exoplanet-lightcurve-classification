from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


NOTEBOOK_DIR = Path(__file__).resolve().parent
ROOT = NOTEBOOK_DIR.parent
RESULTS_DIR = NOTEBOOK_DIR / "resultados"
TABLES_DIR = RESULTS_DIR / "tablas"
FIGURES_DIR = RESULTS_DIR / "figuras"
ACADEMIC_FIGURES_DIR = RESULTS_DIR / "figuras_academicas"
ACADEMIC_ES_DIR = ACADEMIC_FIGURES_DIR / "espanol"
ACADEMIC_EN_DIR = ACADEMIC_FIGURES_DIR / "english"


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    return (-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))).astype(np.float32)


def outcome_labels(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.select(
        [
            (y == 1) & (pred == 1),
            (y == 0) & (pred == 0),
            (y == 0) & (pred == 1),
            (y == 1) & (pred == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="NA",
    )


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def roc_auc_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    sum_ranks_pos = float(ranks[labels].sum())
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 3 or float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 3:
        return float("nan")
    return pearson_corr(rankdata(a), rankdata(b))


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if not np.isfinite(pooled) or pooled <= 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def minmax_by_dataset(df: pd.DataFrame, column: str) -> pd.Series:
    out = pd.Series(index=df.index, dtype=np.float64)
    for dataset, group in df.groupby("dataset"):
        values = group[column].astype(float)
        lo = float(values.min())
        hi = float(values.max())
        if hi - lo <= 1e-12:
            out.loc[group.index] = 0.0
        else:
            out.loc[group.index] = (values - lo) / (hi - lo)
    return out.astype(float)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *rows])


def prepare_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ACADEMIC_ES_DIR.mkdir(parents=True, exist_ok=True)
    ACADEMIC_EN_DIR.mkdir(parents=True, exist_ok=True)


def copy_xai_outputs() -> None:
    source = ROOT / "resultados_xai"
    if not source.exists():
        raise FileNotFoundError(f"No existe {source}. Ejecuta primero xai_mejores_modelos.py.")

    for src in sorted((source / "tablas").glob("*.csv")):
        shutil.copy2(src, TABLES_DIR / src.name)

    for src in sorted((source / "figuras").glob("*.png")):
        shutil.copy2(src, FIGURES_DIR / src.name)


def compute_uncertainty_tables(mc_samples: int = 100, batch_size: int = 512) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sys.path.insert(0, str(ROOT))
    import xai_mejores_modelos as xai

    xai.set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for name in ("kepler", "tess"):
        spec = xai.DATASETS[name]
        model = xai.build_model(ROOT, spec, device)
        xg, xl, y, raw_labels = xai.load_dataset(ROOT, spec)
        threshold = xai.load_threshold(ROOT, spec)

        maxlike_logits = xai.predict_logits(model, xg, xl, device=device, batch_size=batch_size)
        maxlike_prob = xai.sigmoid_np(maxlike_logits)
        maxlike_pred = (maxlike_prob >= threshold).astype(np.int32)

        mc_draws = xai.mc_dropout_probs(
            model,
            xg,
            xl,
            device=device,
            samples=mc_samples,
            batch_size=batch_size,
        )
        mc_mean = mc_draws.mean(axis=0)
        mc_std = mc_draws.std(axis=0)
        mc_pred = (mc_mean >= threshold).astype(np.int32)

        predictive_entropy = binary_entropy(mc_mean)
        expected_entropy = binary_entropy(mc_draws).mean(axis=0)
        epistemic_uncertainty = np.maximum(0.0, predictive_entropy - expected_entropy)
        aleatoric_uncertainty = expected_entropy
        maxlike_entropy = binary_entropy(maxlike_prob)
        target_confidence = np.where(maxlike_pred == 1, maxlike_prob, 1.0 - maxlike_prob)
        outcomes = outcome_labels(y, maxlike_pred)

        for idx in range(len(y)):
            rows.append(
                {
                    "dataset": name,
                    "index": int(idx),
                    "raw_label": raw_labels[idx],
                    "y_true": int(y[idx]),
                    "threshold": float(threshold),
                    "outcome": str(outcomes[idx]),
                    "maxlike_probability": float(maxlike_prob[idx]),
                    "maxlike_prediction": int(maxlike_pred[idx]),
                    "maxlike_confidence": float(target_confidence[idx]),
                    "maxlike_entropy": float(maxlike_entropy[idx]),
                    "mc_dropout_mean_probability": float(mc_mean[idx]),
                    "mc_dropout_std_probability": float(mc_std[idx]),
                    "mc_dropout_prediction": int(mc_pred[idx]),
                    "mc_dropout_predictive_entropy": float(predictive_entropy[idx]),
                    "aleatoric_uncertainty": float(aleatoric_uncertainty[idx]),
                    "epistemic_uncertainty": float(epistemic_uncertainty[idx]),
                    "mc_samples": int(mc_samples),
                }
            )

    uncertainty_df = pd.DataFrame(rows)
    by_outcome = (
        uncertainty_df.groupby(["dataset", "outcome"], as_index=False)
        .agg(
            n=("index", "count"),
            maxlike_confidence_mean=("maxlike_confidence", "mean"),
            maxlike_entropy_mean=("maxlike_entropy", "mean"),
            mc_dropout_std_mean=("mc_dropout_std_probability", "mean"),
            predictive_entropy_mean=("mc_dropout_predictive_entropy", "mean"),
            aleatoric_uncertainty_mean=("aleatoric_uncertainty", "mean"),
            epistemic_uncertainty_mean=("epistemic_uncertainty", "mean"),
        )
        .sort_values(["dataset", "outcome"])
    )
    by_dataset = (
        uncertainty_df.groupby("dataset", as_index=False)
        .agg(
            n=("index", "count"),
            maxlike_confidence_mean=("maxlike_confidence", "mean"),
            maxlike_entropy_mean=("maxlike_entropy", "mean"),
            mc_dropout_std_mean=("mc_dropout_std_probability", "mean"),
            predictive_entropy_mean=("mc_dropout_predictive_entropy", "mean"),
            aleatoric_uncertainty_mean=("aleatoric_uncertainty", "mean"),
            epistemic_uncertainty_mean=("epistemic_uncertainty", "mean"),
        )
        .sort_values("dataset")
    )
    return uncertainty_df, by_outcome, by_dataset


def plot_maxlike_vs_mc(df: pd.DataFrame, dataset: str) -> None:
    sub = df[df["dataset"] == dataset].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    sc = ax.scatter(
        sub["maxlike_probability"],
        sub["mc_dropout_mean_probability"],
        c=sub["epistemic_uncertainty"],
        s=18,
        cmap="magma",
        alpha=0.75,
        edgecolors="none",
    )
    threshold = float(sub["threshold"].iloc[0])
    ax.axvline(threshold, color="#202020", linestyle="--", linewidth=1.0)
    ax.axhline(threshold, color="#202020", linestyle="--", linewidth=1.0)
    ax.plot([0, 1], [0, 1], color="#6c757d", linewidth=0.9, alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("probabilidad MaxLike")
    ax.set_ylabel("probabilidad media MC Dropout")
    ax.set_title(f"{dataset.upper()} - MaxLike vs MC Dropout", fontweight="bold")
    fig.colorbar(sc, ax=ax, label="incertidumbre epistemica")
    fig.savefig(FIGURES_DIR / f"{dataset}_maxlike_vs_mc_dropout.png", dpi=180)
    plt.close(fig)


def plot_uncertainty_by_outcome(df: pd.DataFrame, dataset: str) -> None:
    sub = df[df["dataset"] == dataset].copy()
    if sub.empty:
        return
    order = [x for x in ["TP", "TN", "FP", "FN"] if x in set(sub["outcome"])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    specs = [
        ("aleatoric_uncertainty", "Aleatorica\nE[H(p_t)]"),
        ("epistemic_uncertainty", "Epistemica\nMI"),
        ("mc_dropout_std_probability", "Std MC Dropout"),
    ]
    for ax, (col, title) in zip(axes, specs):
        data = [sub.loc[sub["outcome"] == outcome, col].to_numpy() for outcome in order]
        ax.boxplot(data, tick_labels=order, showmeans=True)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle(f"{dataset.upper()} - Descomposicion de incertidumbre por resultado", fontweight="bold")
    fig.savefig(FIGURES_DIR / f"{dataset}_uncertainty_decomposition_by_outcome.png", dpi=180)
    plt.close(fig)


def compute_faithfulness_auc_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    faith = pd.read_csv(TABLES_DIR / "xai_faithfulness_metrics.csv")
    rows = []
    group_cols = ["dataset", "case", "index", "method", "target_label"]
    for keys, group in faith.groupby(group_cols):
        group = group.sort_values("deleted_fraction")
        fractions = group["deleted_fraction"].to_numpy(dtype=np.float64)
        drops = group["confidence_drop"].to_numpy(dtype=np.float64)
        max_fraction = max(float(fractions.max()), 1e-12)
        diffs = np.diff(drops)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "faithfulness_auc_norm": float(np.trapezoid(drops, fractions) / max_fraction),
                "drop_at_05": float(np.interp(0.05, fractions, drops)),
                "drop_at_10": float(np.interp(0.10, fractions, drops)),
                "drop_at_20": float(np.interp(0.20, fractions, drops)),
                "max_drop": float(np.max(drops)),
                "final_drop": float(drops[-1]),
                "negative_drop_fraction": float(np.mean(drops < -1e-8)),
                "monotonicity_fraction": float(np.mean(diffs >= -1e-8)) if len(diffs) else 1.0,
            }
        )
        rows.append(row)
    sample_df = pd.DataFrame(rows)
    summary_df = (
        sample_df.groupby(["dataset", "method"], as_index=False)
        .agg(
            n=("index", "count"),
            faithfulness_auc_mean=("faithfulness_auc_norm", "mean"),
            faithfulness_auc_std=("faithfulness_auc_norm", "std"),
            drop_at_10_mean=("drop_at_10", "mean"),
            drop_at_20_mean=("drop_at_20", "mean"),
            max_drop_mean=("max_drop", "mean"),
            negative_drop_fraction_mean=("negative_drop_fraction", "mean"),
            monotonicity_fraction_mean=("monotonicity_fraction", "mean"),
        )
        .fillna(0.0)
        .sort_values(["dataset", "faithfulness_auc_mean"], ascending=[True, False])
    )
    return sample_df, summary_df


def plot_faithfulness_auc(summary_df: pd.DataFrame) -> None:
    for dataset, group in summary_df.groupby("dataset"):
        group = group.sort_values("faithfulness_auc_mean")
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.barh(group["method"], group["faithfulness_auc_mean"], color="#457b9d", alpha=0.85)
        ax.set_xlabel("AUC normalizada de caida de confianza")
        ax.set_title(f"{dataset.upper()} - Fidelidad global por metodo", fontweight="bold")
        ax.grid(axis="x", alpha=0.22)
        fig.savefig(FIGURES_DIR / f"{dataset}_faithfulness_auc_by_method.png", dpi=180)
        plt.close(fig)


def target_confidence_batch(
    model: torch.nn.Module,
    xg_batch: np.ndarray,
    xl_batch: np.ndarray,
    target_label: int,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    model.eval()
    confs = []
    with torch.no_grad():
        for start in range(0, len(xg_batch), batch_size):
            end = min(start + batch_size, len(xg_batch))
            bg = torch.tensor(xg_batch[start:end, None, :], dtype=torch.float32, device=device)
            bl = torch.tensor(xl_batch[start:end, None, :], dtype=torch.float32, device=device)
            probs = torch.sigmoid(model(bg, bl)).cpu().numpy().ravel()
            confs.extend((probs if int(target_label) == 1 else 1.0 - probs).tolist())
    return np.asarray(confs, dtype=np.float32)


def random_feature_masks(
    global_len: int,
    local_len: int,
    n_perturbations: int,
    mask_fraction: float,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    kg = max(1, int(round(global_len * mask_fraction)))
    kl = max(1, int(round(local_len * mask_fraction)))
    masks = []
    for _ in range(n_perturbations):
        mg = np.sort(rng.choice(global_len, size=kg, replace=False)).astype(np.int32)
        ml = np.sort(rng.choice(local_len, size=kl, replace=False)).astype(np.int32)
        masks.append((mg, ml))
    return masks


def compute_faithfulness_correlation_tables(
    n_perturbations: int = 40,
    mask_fraction: float = 0.10,
    batch_size: int = 128,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sys.path.insert(0, str(ROOT))
    import xai_mejores_modelos as xai

    examples_path = TABLES_DIR / "xai_examples_summary.csv"
    if not examples_path.exists():
        raise FileNotFoundError(f"No existe {examples_path}. Ejecuta primero xai_mejores_modelos.py.")

    examples = pd.read_csv(examples_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    rng = np.random.default_rng(2026)

    for dataset in ("kepler", "tess"):
        spec = xai.DATASETS[dataset]
        model = xai.build_model(ROOT, spec, device)
        xg_all, xl_all, y_all, raw_labels = xai.load_dataset(ROOT, spec)
        dataset_examples = examples[examples["dataset"] == dataset].copy()

        for _, example in dataset_examples.iterrows():
            idx = int(example["index"])
            case = str(example["case"])
            target_label = int(example["pred_label"])
            xg = xg_all[idx].astype(np.float32)
            xl = xl_all[idx].astype(np.float32)
            base_prob = xai.predict_one_prob(model, xg, xl, device)
            base_conf = float(xai.target_confidence_from_prob(base_prob, target_label))

            sal_g, sal_l = xai.compute_saliency(model, xg, xl, device, target_label)
            ig_g, ig_l = xai.compute_integrated_gradients(model, xg, xl, device, target_label, steps=40)
            mcam_g, mcam_l = xai.compute_multiscale_gradcam(model, xg, xl, device, target_label)
            occ_g, occ_l = xai.compute_occlusion_sensitivity(
                model,
                xg,
                xl,
                device,
                target_label,
                batch_size=batch_size,
            )
            cons_g = xai.consensus_importance(sal_g, ig_g, mcam_g, occ_g)
            cons_l = xai.consensus_importance(sal_l, ig_l, mcam_l, occ_l)
            method_maps = {
                "saliency": (sal_g, sal_l),
                "integrated_gradients": (ig_g, ig_l),
                "gradcam_multiscale": (mcam_g, mcam_l),
                "occlusion": (occ_g, occ_l),
                "consensus": (cons_g, cons_l),
            }

            masks = random_feature_masks(len(xg), len(xl), n_perturbations, mask_fraction, rng)
            perturbed_g = []
            perturbed_l = []
            for mask_g, mask_l in masks:
                gx = xg.copy()
                lx = xl.copy()
                gx[mask_g] = 0.0
                lx[mask_l] = 0.0
                perturbed_g.append(gx)
                perturbed_l.append(lx)
            perturbed_g = np.stack(perturbed_g, axis=0).astype(np.float32)
            perturbed_l = np.stack(perturbed_l, axis=0).astype(np.float32)
            perturbed_conf = target_confidence_batch(
                model,
                perturbed_g,
                perturbed_l,
                target_label,
                device,
                batch_size=batch_size,
            )
            confidence_drops = base_conf - perturbed_conf

            for method, (heat_g, heat_l) in method_maps.items():
                attribution_sums = np.asarray(
                    [float(heat_g[mask_g].sum() + heat_l[mask_l].sum()) for mask_g, mask_l in masks],
                    dtype=np.float64,
                )
                pearson = pearson_corr(attribution_sums, confidence_drops)
                spearman = spearman_corr(attribution_sums, confidence_drops)
                rows.append(
                    {
                        "dataset": dataset,
                        "case": case,
                        "index": idx,
                        "raw_label": raw_labels[idx],
                        "y_true": int(y_all[idx]),
                        "target_label": target_label,
                        "method": method,
                        "base_probability": float(base_prob),
                        "base_target_confidence": float(base_conf),
                        "n_perturbations": int(n_perturbations),
                        "mask_fraction": float(mask_fraction),
                        "global_points_masked": int(round(len(xg) * mask_fraction)),
                        "local_points_masked": int(round(len(xl) * mask_fraction)),
                        "faithfulness_correlation_pearson": pearson,
                        "faithfulness_correlation_spearman": spearman,
                        "mean_attribution_sum": float(np.mean(attribution_sums)),
                        "std_attribution_sum": float(np.std(attribution_sums)),
                        "mean_confidence_drop": float(np.mean(confidence_drops)),
                        "std_confidence_drop": float(np.std(confidence_drops)),
                        "min_confidence_drop": float(np.min(confidence_drops)),
                        "max_confidence_drop": float(np.max(confidence_drops)),
                        "valid_correlation": bool(np.isfinite(pearson) or np.isfinite(spearman)),
                    }
                )

    sample_df = pd.DataFrame(rows)
    summary_df = (
        sample_df.groupby(["dataset", "method"], as_index=False)
        .agg(
            n=("index", "count"),
            valid_n=("valid_correlation", "sum"),
            pearson_mean=("faithfulness_correlation_pearson", "mean"),
            pearson_std=("faithfulness_correlation_pearson", "std"),
            spearman_mean=("faithfulness_correlation_spearman", "mean"),
            spearman_std=("faithfulness_correlation_spearman", "std"),
            mean_confidence_drop=("mean_confidence_drop", "mean"),
            mean_attribution_sum=("mean_attribution_sum", "mean"),
        )
        .fillna(0.0)
        .sort_values(["dataset", "pearson_mean"], ascending=[True, False])
    )
    return sample_df, summary_df


def plot_faithfulness_correlation(summary_df: pd.DataFrame) -> None:
    for dataset, group in summary_df.groupby("dataset"):
        group = group.sort_values("pearson_mean")
        y = np.arange(len(group))
        fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        ax.barh(y - 0.16, group["pearson_mean"], height=0.30, color="#2a9d8f", alpha=0.85, label="Pearson")
        ax.barh(y + 0.16, group["spearman_mean"], height=0.30, color="#457b9d", alpha=0.85, label="Spearman")
        ax.axvline(0.0, color="#202020", linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(group["method"])
        ax.set_xlim(-1.0, 1.0)
        ax.set_xlabel("Faithfulness Correlation")
        ax.set_title(f"{dataset.upper()} - Faithfulness Correlation por metodo", fontweight="bold")
        ax.grid(axis="x", alpha=0.22)
        ax.legend(loc="lower right")
        fig.savefig(FIGURES_DIR / f"{dataset}_faithfulness_correlation_by_method.png", dpi=180)
        plt.close(fig)


def compute_attention_quality_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    focus = pd.read_csv(TABLES_DIR / "xai_attention_focus_by_sample.csv")
    focus = focus.copy()
    focus["is_error"] = focus["outcome"].isin(["FP", "FN"]).astype(int)
    focus["central_norm"] = minmax_by_dataset(focus, "mean_central_importance_ratio")
    focus["variance_norm"] = minmax_by_dataset(focus, "mean_attention_position_variance")
    focus["entropy_norm"] = minmax_by_dataset(focus, "mean_attention_entropy")
    focus["focus_count_norm"] = minmax_by_dataset(focus, "total_focus_count")
    focus["mc_std_norm"] = minmax_by_dataset(focus, "mc_std")
    focus["confidence_inv_norm"] = minmax_by_dataset(focus.assign(confidence_inv=1.0 - focus["target_confidence"]), "confidence_inv")
    focus["attention_quality_score"] = (
        focus["central_norm"]
        + (1.0 - focus["variance_norm"])
        + (1.0 - focus["entropy_norm"])
        + (1.0 - focus["focus_count_norm"])
    ) / 4.0
    focus["attention_risk_score"] = (
        (1.0 - focus["central_norm"])
        + focus["variance_norm"]
        + focus["entropy_norm"]
        + focus["focus_count_norm"]
        + focus["mc_std_norm"]
        + focus["confidence_inv_norm"]
    ) / 6.0
    summary = (
        focus.groupby(["dataset", "outcome"], as_index=False)
        .agg(
            n=("index", "count"),
            attention_quality_mean=("attention_quality_score", "mean"),
            attention_quality_std=("attention_quality_score", "std"),
            attention_risk_mean=("attention_risk_score", "mean"),
            attention_risk_std=("attention_risk_score", "std"),
            total_focus_count_mean=("total_focus_count", "mean"),
            attention_position_variance_mean=("mean_attention_position_variance", "mean"),
            attention_entropy_mean=("mean_attention_entropy", "mean"),
            central_importance_ratio_mean=("mean_central_importance_ratio", "mean"),
        )
        .fillna(0.0)
        .sort_values(["dataset", "outcome"])
    )
    return focus, summary


def plot_attention_quality(quality_df: pd.DataFrame) -> None:
    for dataset, group in quality_df.groupby("dataset"):
        order = [x for x in ["TP", "TN", "FP", "FN"] if x in set(group["outcome"])]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for ax, col, title in [
            (axes[0], "attention_quality_score", "Quality score"),
            (axes[1], "attention_risk_score", "Risk score"),
        ]:
            data = [group.loc[group["outcome"] == outcome, col].to_numpy() for outcome in order]
            ax.boxplot(data, tick_labels=order, showmeans=True)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.22)
        fig.suptitle(f"{dataset.upper()} - Scores derivados de XAI por outcome", fontweight="bold")
        fig.savefig(FIGURES_DIR / f"{dataset}_attention_quality_risk_scores.png", dpi=180)
        plt.close(fig)


def language_config(lang: str) -> dict:
    if lang == "en":
        return {
            "outcome_labels": {"TN": "TN", "FP": "FP", "FN": "FN", "TP": "TP"},
            "x_pred": "Model prediction",
            "y_true": "True label",
            "pred_ticks": ["Predicted 0", "Predicted 1"],
            "true_ticks": ["True 0", "True 1"],
            "conf_title": "Row-Normalized Confusion Matrix with XAI Attention Summary",
            "conf_cbar": "Row-normalized proportion",
            "cell_rate": "of true",
            "focus": "focus",
            "disp": "disp.",
            "note": "Color encodes row-normalized class proportion; text reports count and attention-focus statistics.",
            "risk_title": "XAI Attention Risk Score by Prediction Outcome",
            "risk_ylabel": "Attention risk score",
            "risk_note": "Higher values indicate lower centrality, higher dispersion, more foci, higher uncertainty or lower confidence.",
            "focus_title": "Attention Focus Count and Temporal Dispersion by Outcome",
            "focus_ylabel": "Number of attention foci",
            "var_ylabel": "Attention position variance",
            "focus_panel": "Attention foci",
        "var_panel": "Temporal dispersion",
        "outcome": "Outcome",
        "mean": "mean",
        "variance": "var.",
        }
    return {
        "outcome_labels": {"TN": "VN", "FP": "FP", "FN": "FN", "TP": "VP"},
        "x_pred": "Prediccion del modelo",
        "y_true": "Etiqueta real",
        "pred_ticks": ["Predicho 0", "Predicho 1"],
        "true_ticks": ["Real 0", "Real 1"],
        "conf_title": "Matriz de confusion normalizada con resumen de atencion XAI",
        "conf_cbar": "Proporcion normalizada por clase real",
        "cell_rate": "de real",
        "focus": "focos",
        "disp": "disp.",
        "note": "El color codifica la proporcion normalizada por fila; el texto reporta conteo y estadisticos de atencion.",
        "risk_title": "Score de riesgo de atencion XAI por tipo de prediccion",
        "risk_ylabel": "Score de riesgo de atencion",
        "risk_note": "Valores mayores indican menor centralidad, mayor dispersion, mas focos, mayor incertidumbre o menor confianza.",
        "focus_title": "Cantidad de focos y dispersion temporal de atencion por resultado",
        "focus_ylabel": "Numero de focos de atencion",
        "var_ylabel": "Varianza de posicion de atencion",
        "focus_panel": "Focos de atencion",
        "var_panel": "Dispersion temporal",
        "outcome": "Resultado",
        "mean": "media",
        "variance": "var.",
    }


def academic_filename(dataset: str, kind: str, lang: str) -> str:
    if lang == "en":
        names = {
            "confusion": f"{dataset}_row_normalized_confusion_matrix_xai_attention.png",
            "risk": f"{dataset}_xai_attention_risk_score_violin.png",
            "focus_variance": f"{dataset}_attention_focus_count_and_variance_violin.png",
        }
    else:
        names = {
            "confusion": f"{dataset}_matriz_confusion_normalizada_atencion_xai.png",
            "risk": f"{dataset}_score_riesgo_atencion_xai_violin.png",
            "focus_variance": f"{dataset}_focos_atencion_y_varianza_violin.png",
        }
    return names[kind]


def ensure_confusion_normalization(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "row_normalized_rate" not in df.columns:
        row_totals = df.groupby(["dataset", "y_true"])["n"].transform("sum").replace(0, np.nan)
        dataset_totals = df.groupby("dataset")["n"].transform("sum").replace(0, np.nan)
        df["row_total"] = row_totals
        df["row_normalized_rate"] = (df["n"] / row_totals).fillna(0.0)
        df["global_rate"] = (df["n"] / dataset_totals).fillna(0.0)
    return df


def plot_academic_confusion_matrix(dataset: str, summary_df: pd.DataFrame, out_path: Path, lang: str) -> None:
    cfg = language_config(lang)
    df = ensure_confusion_normalization(summary_df)
    df = df[df["dataset"] == dataset].copy()
    if df.empty:
        return

    rate_matrix = np.zeros((2, 2), dtype=np.float64)
    labels = [["" for _ in range(2)] for _ in range(2)]
    for row in df.itertuples(index=False):
        y_idx = int(row.y_true)
        p_idx = int(row.pred_label)
        outcome = cfg["outcome_labels"].get(str(row.outcome), str(row.outcome))
        rate = float(row.row_normalized_rate)
        rate_matrix[y_idx, p_idx] = rate
        labels[y_idx][p_idx] = (
            f"{outcome}\n"
            f"{100.0 * rate:.1f}% {cfg['cell_rate']} {y_idx}\n"
            f"n={int(row.n)}\n"
            f"{cfg['focus']}={row.total_focus_count_mean:.2f}+/-{row.total_focus_count_std:.2f}\n"
            f"{cfg['disp']}={row.attention_position_variance_mean:.3f}"
        )

    fig, ax = plt.subplots(figsize=(8.8, 7.0), constrained_layout=True)
    im = ax.imshow(rate_matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(cfg["pred_ticks"])
    ax.set_yticklabels(cfg["true_ticks"])
    ax.set_xlabel(cfg["x_pred"])
    ax.set_ylabel(cfg["y_true"])
    ax.set_title(f"{dataset.upper()} - {cfg['conf_title']}", fontweight="bold")
    for i in range(2):
        for j in range(2):
            color = "white" if rate_matrix[i, j] >= 0.55 else "#202020"
            ax.text(j, i, labels[i][j] or "n=0", ha="center", va="center", fontsize=9.4, color=color)
    fig.colorbar(im, ax=ax, label=cfg["conf_cbar"])
    ax.text(0.5, -0.15, cfg["note"], ha="center", va="center", transform=ax.transAxes, fontsize=8.8, color="#404040")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def violin_with_stats(
    ax: plt.Axes,
    values_by_group: list[np.ndarray],
    labels: list[str],
    ylabel: str,
    title: str,
    cfg: dict,
    colors: list[str],
) -> None:
    positions = np.arange(1, len(values_by_group) + 1)
    non_empty = [np.asarray(v, dtype=np.float64) for v in values_by_group if len(v)]
    if not non_empty:
        return
    all_values = np.concatenate(non_empty)
    global_range = max(float(np.nanmax(all_values)) - float(np.nanmin(all_values)), 1e-6)
    data_min = float(np.nanmin(all_values))
    data_max = float(np.nanmax(all_values))
    ax.set_ylim(data_min - 0.08 * global_range, data_max + 0.26 * global_range)

    def fmt_stat(value: float) -> str:
        value = float(value)
        if value != 0.0 and abs(value) < 1e-3:
            return f"{value:.2e}"
        return f"{value:.3f}"

    parts = ax.violinplot(values_by_group, positions=positions, widths=0.78, showmeans=False, showmedians=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#263238")
        body.set_alpha(0.72)
        body.set_linewidth(0.9)
    for key in ["cbars", "cmins", "cmaxes"]:
        parts[key].set_color("#37474f")
        parts[key].set_linewidth(0.8)

    rng = np.random.default_rng(42)
    for pos, vals, color in zip(positions, values_by_group, colors):
        vals = np.asarray(vals, dtype=np.float64)
        if vals.size == 0:
            continue
        sample = vals if vals.size <= 240 else rng.choice(vals, size=240, replace=False)
        jitter = rng.normal(0.0, 0.045, size=len(sample))
        ax.scatter(np.full(len(sample), pos) + jitter, sample, s=7, color="#111111", alpha=0.18, linewidths=0)

        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        mean = float(np.mean(vals))
        var = float(np.var(vals, ddof=1)) if vals.size > 1 else 0.0
        ax.vlines(pos, q1, q3, color="#111111", linewidth=3.0, alpha=0.65)
        ax.scatter([pos], [mean], s=46, color="white", edgecolor="#111111", linewidth=1.1, zorder=4)
        ax.hlines(med, pos - 0.17, pos + 0.17, color="#111111", linewidth=1.2, zorder=4)
        y_offset = 0.035 * global_range
        ax.text(
            pos,
            float(np.nanmax(vals)) + y_offset,
            f"{cfg['mean']}={fmt_stat(mean)}\n{cfg['variance']}={fmt_stat(var)}",
            ha="center",
            va="bottom",
            fontsize=7.8,
            color="#263238",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel(cfg["outcome"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.22)


def plot_academic_attention_risk_violin(dataset: str, quality_df: pd.DataFrame, out_path: Path, lang: str) -> None:
    cfg = language_config(lang)
    df = quality_df[quality_df["dataset"] == dataset].copy()
    if df.empty or "attention_risk_score" not in df.columns:
        return
    order = [x for x in ["TN", "FP", "FN", "TP"] if x in set(df["outcome"])]
    labels = [cfg["outcome_labels"].get(x, x) for x in order]
    values = [df.loc[df["outcome"] == outcome, "attention_risk_score"].dropna().to_numpy() for outcome in order]
    colors = ["#3d5a80", "#e07a5f", "#c1121f", "#2a9d8f"][: len(order)]

    fig, ax = plt.subplots(figsize=(9.5, 6.2), constrained_layout=True)
    violin_with_stats(ax, values, labels, cfg["risk_ylabel"], f"{dataset.upper()} - {cfg['risk_title']}", cfg, colors)
    ax.text(0.5, -0.18, cfg["risk_note"], ha="center", va="center", transform=ax.transAxes, fontsize=8.8, color="#404040")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_academic_focus_variance_violin(dataset: str, focus_df: pd.DataFrame, out_path: Path, lang: str) -> None:
    cfg = language_config(lang)
    df = focus_df[focus_df["dataset"] == dataset].copy()
    if df.empty:
        return
    order = [x for x in ["TN", "FP", "FN", "TP"] if x in set(df["outcome"])]
    labels = [cfg["outcome_labels"].get(x, x) for x in order]
    colors = ["#3d5a80", "#e07a5f", "#c1121f", "#2a9d8f"][: len(order)]

    focus_values = [df.loc[df["outcome"] == outcome, "total_focus_count"].dropna().to_numpy() for outcome in order]
    variance_values = [
        df.loc[df["outcome"] == outcome, "mean_attention_position_variance"].dropna().to_numpy()
        for outcome in order
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), constrained_layout=True)
    violin_with_stats(
        axes[0],
        focus_values,
        labels,
        cfg["focus_ylabel"],
        cfg["focus_panel"],
        cfg,
        colors,
    )
    violin_with_stats(
        axes[1],
        variance_values,
        labels,
        cfg["var_ylabel"],
        cfg["var_panel"],
        cfg,
        colors,
    )
    fig.suptitle(f"{dataset.upper()} - {cfg['focus_title']}", fontweight="bold", fontsize=13)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_academic_figures_readme() -> None:
    text = """# Figuras academicas XAI

Esta carpeta contiene dos versiones de las figuras listas para informe o paper:

- `espanol/`: nombres, titulos y organizacion en espanol.
- `english/`: nombres, titulos y organizacion en ingles.

## Estructura

```text
figuras_academicas/
├── indice_figuras_academicas.csv
├── analisis.txt
├── conclusión_final.txt
├── figuras_academicas_explicacion_unaAuna.txt
├── fidelidad_ampliada/
├── espanol/
│   ├── figuras_principales/
│   └── material_suplementario/
└── english/
    ├── main_figures/
    └── supplementary_figures/
```

## Figuras principales

Las carpetas `figuras_principales/` y `main_figures/` contienen las figuras mas recomendables para el cuerpo del informe/paper:

- matriz de confusion normalizada por fila real con resumen de focos XAI;
- violin plot del score de riesgo de atencion XAI por resultado;
- violin plot de cantidad de focos y varianza/dispersion temporal de atencion por resultado.

Estas figuras fueron regeneradas con mayor resolucion y con rotulos mas formales.

## Material suplementario

Las carpetas `material_suplementario/` y `supplementary_figures/` contienen todas las figuras generadas durante el analisis XAI, copiadas con nombres academicos en espanol e ingles. Esto incluye mapas XAI individuales, curvas de fidelidad, calibracion, incertidumbre, contrafactuales, centralidad, error detection y comparaciones MaxLike/MC Dropout.

`indice_figuras_academicas.csv` permite rastrear cada figura desde su archivo original hasta su version academica en ambos idiomas.

## Analisis general

El archivo `analisis.txt` entrega una lectura global de los resultados XAI. Resume desempeno predictivo, matrices normalizadas, comportamiento de los focos de atencion, incertidumbre MaxLike/MC Dropout, fidelidad, contrafactuales, calibracion, limitaciones y conclusiones generales.

## Conclusion final

El archivo `conclusión_final.txt` contiene una seccion lista para informe. Responde explicitamente que se logro explicar, si las explicaciones ayudan a interpretar el modelo, por que ocurren, como podrian mejorar el sistema de ML y como se interpreta la fidelidad de las explicaciones.

## Fidelidad ampliada

La carpeta `fidelidad_ampliada/` contiene una evaluacion tipo Faithfulness Correlation sobre 260 muestras estratificadas del test set. Esta carpeta documenta que Quantus no fue usado literalmente, explica por que SHAP se elimina del paper, y entrega tablas/figuras nuevas para defender mejor la fidelidad XAI.

## Explicacion una a una

El archivo `figuras_academicas_explicacion_unaAuna.txt` explica cada figura principal y suplementaria, indicando que muestra, como interpretarla y cual es la conclusion especifica para Kepler o TESS.
"""
    (ACADEMIC_FIGURES_DIR / "README.md").write_text(text, encoding="utf-8")


def clean_png_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for old in path.glob("*.png"):
        old.unlink()


def figure_name_metadata(filename: str) -> dict:
    stem = Path(filename).stem
    dataset = stem.split("_", 1)[0] if "_" in stem else "general"
    rest = stem.split("_", 1)[1] if "_" in stem else stem

    case_map = {
        "positivo_correcto_alta_confianza": (
            "mapa_xai_verdadero_positivo_alta_confianza",
            "xai_explanation_map_high_confidence_true_positive",
            "Mapa XAI de verdadero positivo de alta confianza",
            "XAI explanation map for high-confidence true positive",
        ),
        "negativo_correcto_alta_confianza": (
            "mapa_xai_verdadero_negativo_alta_confianza",
            "xai_explanation_map_high_confidence_true_negative",
            "Mapa XAI de verdadero negativo de alta confianza",
            "XAI explanation map for high-confidence true negative",
        ),
        "cerca_del_umbral": (
            "mapa_xai_ejemplo_cerca_umbral_decision",
            "xai_explanation_map_near_decision_threshold",
            "Mapa XAI de ejemplo cercano al umbral de decision",
            "XAI explanation map for near-threshold example",
        ),
        "falso_positivo": (
            "mapa_xai_falso_positivo",
            "xai_explanation_map_false_positive",
            "Mapa XAI de falso positivo",
            "XAI explanation map for false positive",
        ),
        "falso_negativo": (
            "mapa_xai_falso_negativo",
            "xai_explanation_map_false_negative",
            "Mapa XAI de falso negativo",
            "XAI explanation map for false negative",
        ),
        "mayor_incertidumbre_mc_dropout": (
            "mapa_xai_mayor_incertidumbre_mc_dropout",
            "xai_explanation_map_highest_mc_dropout_uncertainty",
            "Mapa XAI del ejemplo con mayor incertidumbre MC Dropout",
            "XAI explanation map for highest MC Dropout uncertainty",
        ),
    }
    for case, (name_es, name_en, title_es, title_en) in case_map.items():
        if rest.startswith(case):
            idx = rest.split("_idx")[-1].replace("_xai_maps", "") if "_idx" in rest else "sin_idx"
            return {
                "dataset": dataset,
                "category": "mapas_xai_individuales",
                "spanish_stem": f"{dataset}_{name_es}_idx{idx}",
                "english_stem": f"{dataset}_{name_en}_idx{idx}",
                "title_es": f"{dataset.upper()} - {title_es} (idx={idx})",
                "title_en": f"{dataset.upper()} - {title_en} (idx={idx})",
            }

    mapping = {
        "aggregate_importance_selected_examples": (
            "importancia_agregada_ejemplos_seleccionados",
            "aggregate_importance_selected_examples",
            "Importancia agregada en ejemplos seleccionados",
            "Aggregate importance over selected examples",
            "mapas_importancia_agregada",
        ),
        "attention_focus_confusion_matrix": (
            "matriz_confusion_normalizada_resumen_atencion_xai",
            "row_normalized_confusion_matrix_xai_attention_summary",
            "Matriz de confusion normalizada con resumen de atencion XAI",
            "Row-normalized confusion matrix with XAI attention summary",
            "matrices_confusion",
        ),
        "attention_focus_outcome_boxplot": (
            "boxplot_focos_atencion_dispersion_por_resultado",
            "attention_focus_dispersion_boxplot_by_outcome",
            "Boxplot de focos de atencion y dispersion por resultado",
            "Attention-focus and dispersion boxplot by outcome",
            "focos_atencion",
        ),
        "attention_quality_risk_scores": (
            "scores_calidad_riesgo_atencion_xai",
            "xai_attention_quality_and_risk_scores",
            "Scores de calidad y riesgo de atencion XAI",
            "XAI attention quality and risk scores",
            "riesgo_atencion",
        ),
        "counterfactual_effects": (
            "analisis_contrafactual_efecto_probabilidad",
            "counterfactual_probability_effect_analysis",
            "Analisis contrafactual del efecto en probabilidad",
            "Counterfactual probability-effect analysis",
            "contrafactuales",
        ),
        "faithfulness_auc_by_method": (
            "fidelidad_auc_borrado_por_metodo_xai",
            "faithfulness_deletion_auc_by_xai_method",
            "Fidelidad por AUC de borrado segun metodo XAI",
            "Deletion-AUC faithfulness by XAI method",
            "fidelidad_xai",
        ),
        "faithfulness_correlation_by_method": (
            "correlacion_fidelidad_por_metodo_xai",
            "faithfulness_correlation_by_xai_method",
            "Faithfulness Correlation por metodo XAI",
            "Faithfulness Correlation by XAI method",
            "fidelidad_xai",
        ),
        "faithfulness_deletion_curves": (
            "curvas_fidelidad_por_borrado",
            "faithfulness_deletion_curves",
            "Curvas de fidelidad por borrado",
            "Faithfulness deletion curves",
            "fidelidad_xai",
        ),
        "maxlike_vs_mc_dropout": (
            "comparacion_maxlike_mc_dropout",
            "maxlike_vs_mc_dropout_probability_comparison",
            "Comparacion de probabilidad MaxLike y MC Dropout",
            "MaxLike vs MC Dropout probability comparison",
            "incertidumbre",
        ),
        "mc_dropout_scatter": (
            "dispersion_probabilidad_incertidumbre_mc_dropout",
            "mc_dropout_probability_uncertainty_scatter",
            "Probabilidad vs incertidumbre MC Dropout",
            "MC Dropout probability vs uncertainty scatter",
            "incertidumbre",
        ),
        "mc_dropout_uncertainty": (
            "incertidumbre_mc_dropout_ejemplos_seleccionados",
            "mc_dropout_uncertainty_selected_examples",
            "Incertidumbre MC Dropout en ejemplos seleccionados",
            "MC Dropout uncertainty over selected examples",
            "incertidumbre",
        ),
        "outcome_uncertainty_summary": (
            "resumen_incertidumbre_por_resultado",
            "uncertainty_summary_by_prediction_outcome",
            "Resumen de incertidumbre por resultado de prediccion",
            "Uncertainty summary by prediction outcome",
            "incertidumbre",
        ),
        "reliability_calibration": (
            "diagrama_confiabilidad_calibracion_probabilistica",
            "probability_calibration_reliability_diagram",
            "Diagrama de confiabilidad y calibracion probabilistica",
            "Probability calibration reliability diagram",
            "calibracion",
        ),
        "uncertainty_decomposition_by_outcome": (
            "descomposicion_incertidumbre_por_resultado",
            "uncertainty_decomposition_by_prediction_outcome",
            "Descomposicion de incertidumbre por resultado",
            "Uncertainty decomposition by prediction outcome",
            "incertidumbre",
        ),
        "xai_error_detection_auc": (
            "auc_deteccion_errores_metricas_xai",
            "error_detection_auc_from_xai_metrics",
            "AUC de deteccion de errores usando metricas XAI",
            "Error-detection AUC from XAI metrics",
            "deteccion_errores",
        ),
        "xai_method_centrality": (
            "centralidad_importancia_por_metodo_xai",
            "central_importance_ratio_by_xai_method",
            "Centralidad de importancia por metodo XAI",
            "Central importance ratio by XAI method",
            "centralidad_xai",
        ),
    }
    if rest in mapping:
        name_es, name_en, title_es, title_en, category = mapping[rest]
        return {
            "dataset": dataset,
            "category": category,
            "spanish_stem": f"{dataset}_{name_es}",
            "english_stem": f"{dataset}_{name_en}",
            "title_es": f"{dataset.upper()} - {title_es}",
            "title_en": f"{dataset.upper()} - {title_en}",
        }

    safe_rest = rest.replace("__", "_")
    return {
        "dataset": dataset,
        "category": "otras_figuras_xai",
        "spanish_stem": f"{dataset}_figura_xai_{safe_rest}",
        "english_stem": f"{dataset}_xai_figure_{safe_rest}",
        "title_es": f"{dataset.upper()} - Figura XAI: {safe_rest.replace('_', ' ')}",
        "title_en": f"{dataset.upper()} - XAI figure: {safe_rest.replace('_', ' ')}",
    }


def main_figure_catalog() -> list[dict]:
    return [
        {
            "dataset": dataset,
            "kind": "confusion",
            "spanish_source": ACADEMIC_ES_DIR / academic_filename(dataset, "confusion", "es"),
            "english_source": ACADEMIC_EN_DIR / academic_filename(dataset, "confusion", "en"),
            "spanish_name": f"{dataset}_figura_principal_matriz_confusion_normalizada_atencion_xai.png",
            "english_name": f"{dataset}_main_figure_row_normalized_confusion_matrix_xai_attention.png",
            "title_es": f"{dataset.upper()} - Matriz de confusion normalizada con resumen XAI",
            "title_en": f"{dataset.upper()} - Row-normalized confusion matrix with XAI summary",
        }
        for dataset in ("kepler", "tess")
    ] + [
        {
            "dataset": dataset,
            "kind": "risk_violin",
            "spanish_source": ACADEMIC_ES_DIR / academic_filename(dataset, "risk", "es"),
            "english_source": ACADEMIC_EN_DIR / academic_filename(dataset, "risk", "en"),
            "spanish_name": f"{dataset}_figura_principal_violin_score_riesgo_atencion_xai.png",
            "english_name": f"{dataset}_main_figure_xai_attention_risk_score_violin.png",
            "title_es": f"{dataset.upper()} - Violin plot del score de riesgo de atencion XAI",
            "title_en": f"{dataset.upper()} - Violin plot of XAI attention risk score",
        }
        for dataset in ("kepler", "tess")
    ] + [
        {
            "dataset": dataset,
            "kind": "focus_variance_violin",
            "spanish_source": ACADEMIC_ES_DIR / academic_filename(dataset, "focus_variance", "es"),
            "english_source": ACADEMIC_EN_DIR / academic_filename(dataset, "focus_variance", "en"),
            "spanish_name": f"{dataset}_figura_principal_violin_focos_y_varianza_atencion.png",
            "english_name": f"{dataset}_main_figure_attention_focus_count_and_variance_violin.png",
            "title_es": f"{dataset.upper()} - Violin plot de focos y varianza de atencion",
            "title_en": f"{dataset.upper()} - Violin plot of attention foci and attention variance",
        }
        for dataset in ("kepler", "tess")
    ]


def export_paper_ready_academic_catalog() -> None:
    es_main = ACADEMIC_ES_DIR / "figuras_principales"
    en_main = ACADEMIC_EN_DIR / "main_figures"
    es_supp = ACADEMIC_ES_DIR / "material_suplementario"
    en_supp = ACADEMIC_EN_DIR / "supplementary_figures"
    for directory in [es_main, en_main, es_supp, en_supp]:
        clean_png_dir(directory)

    rows = []
    for item in main_figure_catalog():
        if item["spanish_source"].exists():
            dst = es_main / item["spanish_name"]
            shutil.copy2(item["spanish_source"], dst)
            rows.append(
                {
                    "role": "main_paper",
                    "dataset": item["dataset"],
                    "category": item["kind"],
                    "language": "espanol",
                    "original_file": item["spanish_source"].name,
                    "academic_file": str(dst.relative_to(ACADEMIC_FIGURES_DIR)),
                    "title": item["title_es"],
                }
            )
        if item["english_source"].exists():
            dst = en_main / item["english_name"]
            shutil.copy2(item["english_source"], dst)
            rows.append(
                {
                    "role": "main_paper",
                    "dataset": item["dataset"],
                    "category": item["kind"],
                    "language": "english",
                    "original_file": item["english_source"].name,
                    "academic_file": str(dst.relative_to(ACADEMIC_FIGURES_DIR)),
                    "title": item["title_en"],
                }
            )

    for src in sorted(FIGURES_DIR.glob("*.png")):
        meta = figure_name_metadata(src.name)
        es_name = f"{meta['spanish_stem']}.png"
        en_name = f"{meta['english_stem']}.png"
        es_dst = es_supp / es_name
        en_dst = en_supp / en_name
        shutil.copy2(src, es_dst)
        shutil.copy2(src, en_dst)
        rows.append(
            {
                "role": "supplementary",
                "dataset": meta["dataset"],
                "category": meta["category"],
                "language": "espanol",
                "original_file": src.name,
                "academic_file": str(es_dst.relative_to(ACADEMIC_FIGURES_DIR)),
                "title": meta["title_es"],
            }
        )
        rows.append(
            {
                "role": "supplementary",
                "dataset": meta["dataset"],
                "category": meta["category"],
                "language": "english",
                "original_file": src.name,
                "academic_file": str(en_dst.relative_to(ACADEMIC_FIGURES_DIR)),
                "title": meta["title_en"],
            }
        )

    pd.DataFrame(rows).to_csv(ACADEMIC_FIGURES_DIR / "indice_figuras_academicas.csv", index=False)


def generate_academic_figure_sets(quality_df: pd.DataFrame) -> None:
    confusion_path = TABLES_DIR / "xai_attention_focus_confusion_matrix.csv"
    if not confusion_path.exists():
        return
    confusion_df = ensure_confusion_normalization(pd.read_csv(confusion_path))
    confusion_df.to_csv(confusion_path, index=False)

    for lang, out_dir in [("es", ACADEMIC_ES_DIR), ("en", ACADEMIC_EN_DIR)]:
        for dataset in sorted(quality_df["dataset"].dropna().unique()):
            plot_academic_confusion_matrix(
                dataset,
                confusion_df,
                out_dir / academic_filename(dataset, "confusion", lang),
                lang,
            )
            plot_academic_attention_risk_violin(
                dataset,
                quality_df,
                out_dir / academic_filename(dataset, "risk", lang),
                lang,
            )
            plot_academic_focus_variance_violin(
                dataset,
                quality_df,
                out_dir / academic_filename(dataset, "focus_variance", lang),
                lang,
            )
    export_paper_ready_academic_catalog()
    write_academic_figures_readme()


def compute_error_detection_metrics(quality_df: pd.DataFrame, uncertainty_df: pd.DataFrame) -> pd.DataFrame:
    merged = quality_df.merge(
        uncertainty_df[
            [
                "dataset",
                "index",
                "maxlike_entropy",
                "mc_dropout_std_probability",
                "aleatoric_uncertainty",
                "epistemic_uncertainty",
            ]
        ],
        on=["dataset", "index"],
        how="left",
    )
    metrics = [
        "total_focus_count",
        "mean_attention_position_variance",
        "mean_attention_entropy",
        "mean_central_importance_ratio",
        "attention_quality_score",
        "attention_risk_score",
        "mc_std",
        "target_confidence",
        "maxlike_entropy",
        "mc_dropout_std_probability",
        "aleatoric_uncertainty",
        "epistemic_uncertainty",
    ]
    rows = []
    for dataset, group in merged.groupby("dataset"):
        labels = group["is_error"].to_numpy(dtype=bool)
        for metric in metrics:
            scores = group[metric].to_numpy(dtype=np.float64)
            auc_high = roc_auc_from_scores(labels, scores)
            auc_best = max(auc_high, 1.0 - auc_high) if np.isfinite(auc_high) else float("nan")
            direction = "high=risk" if np.isfinite(auc_high) and auc_high >= 0.5 else "low=risk"
            error_values = scores[labels]
            correct_values = scores[~labels]
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "n_error": int(labels.sum()),
                    "n_correct": int((~labels).sum()),
                    "mean_error": float(np.mean(error_values)) if len(error_values) else np.nan,
                    "mean_correct": float(np.mean(correct_values)) if len(correct_values) else np.nan,
                    "cohen_d_error_minus_correct": cohen_d(error_values, correct_values),
                    "roc_auc_error_when_high": auc_high,
                    "roc_auc_best_direction": auc_best,
                    "risk_direction": direction,
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "roc_auc_best_direction"], ascending=[True, False])


def plot_error_detection_metrics(error_df: pd.DataFrame) -> None:
    for dataset, group in error_df.groupby("dataset"):
        top = group.sort_values("roc_auc_best_direction", ascending=False).head(10).sort_values("roc_auc_best_direction")
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        ax.barh(top["metric"], top["roc_auc_best_direction"], color="#e76f51", alpha=0.85)
        ax.axvline(0.5, color="#202020", linestyle="--", linewidth=0.9)
        ax.set_xlim(0.45, 1.0)
        ax.set_xlabel("ROC-AUC para separar errores vs aciertos")
        ax.set_title(f"{dataset.upper()} - Potencial de metricas XAI para detectar errores", fontweight="bold")
        ax.grid(axis="x", alpha=0.22)
        fig.savefig(FIGURES_DIR / f"{dataset}_xai_error_detection_auc.png", dpi=180)
        plt.close(fig)


def compute_method_agreement_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    imp = pd.read_csv(TABLES_DIR / "xai_importance_metrics.csv")
    overlap_cols = [c for c in imp.columns if c.startswith("top10_overlap")]
    central_cols = [c for c in imp.columns if c.endswith("_central_importance_ratio")]
    agreement = (
        imp.groupby(["dataset", "view"], as_index=False)[overlap_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    agreement.columns = ["_".join([str(x) for x in col if str(x)]) for col in agreement.columns.to_flat_index()]
    central = (
        imp.groupby(["dataset", "view"], as_index=False)[central_cols]
        .mean()
        .sort_values(["dataset", "view"])
    )
    return agreement, central


def plot_method_centrality(central_df: pd.DataFrame) -> None:
    for dataset, group in central_df.groupby("dataset"):
        value_cols = [c for c in group.columns if c.endswith("_central_importance_ratio")]
        plot_df = group.set_index("view")[value_cols].T.astype(float)
        plot_df.index = [idx.replace("_central_importance_ratio", "") for idx in plot_df.index]
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        plot_df.plot(kind="bar", ax=ax)
        ax.set_ylabel("fraccion de importancia central")
        ax.set_title(f"{dataset.upper()} - Centralidad por metodo XAI", fontweight="bold")
        ax.grid(axis="y", alpha=0.22)
        ax.legend(title="vista")
        fig.savefig(FIGURES_DIR / f"{dataset}_xai_method_centrality.png", dpi=180)
        plt.close(fig)


def write_readme(faith_corr_summary_df: pd.DataFrame | None = None) -> None:
    if faith_corr_summary_df is None or faith_corr_summary_df.empty:
        faith_corr_table = "No se encontraron resultados de Faithfulness Correlation."
    else:
        cols = ["dataset", "method", "valid_n", "pearson_mean", "spearman_mean", "mean_confidence_drop"]
        faith_corr_table = dataframe_to_markdown(faith_corr_summary_df[cols].round(4))

    text = """# Notebook de resultados XAI

Esta carpeta concentra el notebook principal y una copia ordenada de los resultados usados para discutir interpretabilidad.

## Archivos principales

- `xai_resultados_avanzados.ipynb`: notebook explicativo con tablas y figuras.
- `generar_notebook_resultados.py`: script para reconstruir `resultados/`.
- `resultados/tablas/`: tablas CSV copiadas desde `resultados_xai` y tablas nuevas de incertidumbre.
- `resultados/figuras/`: figuras XAI, matrices de focos y graficos de incertidumbre.
- `resultados/figuras_academicas/espanol/`: figuras con titulos y ejes academicos en espanol.
- `resultados/figuras_academicas/english/`: figuras equivalentes en ingles.
- `resultados/figuras_academicas/analisis.txt`: analisis general de desempeno, incertidumbre, fidelidad, focos de atencion y conclusiones.
- `resultados/figuras_academicas/conclusión_final.txt`: conclusion final lista para informe, con discusion de XAI y fidelidad.
- `resultados/figuras_academicas/figuras_academicas_explicacion_unaAuna.txt`: explicacion individual de cada figura principal y suplementaria.
- `resultados/figuras_academicas/fidelidad_ampliada/`: evaluacion ampliada tipo Faithfulness Correlation sobre 260 muestras estratificadas; incluye nota sobre Quantus y SHAP.

## Incertidumbre

`MaxLike` corresponde a la inferencia deterministica del modelo con `model.eval()` y dropout desactivado. Es la prediccion puntual de maxima verosimilitud usada como base.

`MC Dropout` ejecuta multiples inferencias con dropout activo. La media de esas predicciones resume la probabilidad predictiva y la desviacion estandar mide variabilidad entre muestras.

Para clasificacion binaria se usa:

- Incertidumbre predictiva: `H(E[p])`.
- Incertidumbre aleatorica: `E[H(p_t)]`.
- Incertidumbre epistemica: `H(E[p]) - E[H(p_t)]`.

La incertidumbre aleatorica representa ruido/inherente ambiguedad de los datos. La epistemica representa incertidumbre del modelo y deberia bajar con mas datos/modelos mejores.

## Focos de atencion

El conteo de focos usa Saliency target-aware en todo el test set. Se cuentan componentes conectados de alta relevancia en las vistas global y local. Es un analisis exploratorio para discutir si errores y aciertos presentan distinta cantidad o dispersion de zonas de atencion.

Las matrices de confusion de focos se reportan normalizadas por fila real. Es decir, cada fila suma 100% y el color representa la proporcion de ejemplos de una etiqueta real que cae en cada prediccion. El texto de cada celda mantiene el conteo `n`, la media de focos y la dispersion temporal de la atencion.

Tambien se generan violin plots academicos. El primer violin plot resume el `attention_risk_score`, que combina centralidad, dispersion, cantidad de focos, incertidumbre y confianza. El segundo muestra la distribucion de la cantidad de focos y la varianza de posicion de la atencion, anotando media y varianza por grupo.

## Metricas XAI adicionales

Ademas de las figuras originales, se generan metricas derivadas:

- `xai_faithfulness_auc_summary.csv`: AUC normalizada de la caida de confianza en la prueba de borrado.
- `xai_faithfulness_correlation_summary.csv`: Faithfulness Correlation estilo Bhatt et al. sobre perturbaciones aleatorias.
- `xai_method_agreement_summary.csv`: acuerdo top-10 entre metodos de explicabilidad.
- `xai_method_centrality_summary.csv`: fraccion de importancia central por metodo y vista.
- `xai_attention_quality_summary.csv`: score de calidad/riesgo de atencion por TP/TN/FP/FN.
- `xai_error_detection_metrics.csv`: ROC-AUC de metricas XAI/incertidumbre para separar errores de aciertos.

Estas metricas son utiles para comparar metodos y plantear trabajo futuro, por ejemplo usar estadisticos XAI como features de alerta de predicciones poco confiables.

## Faithfulness Correlation

Se agrego una implementacion local de Faithfulness Correlation, equivalente en objetivo a la metrica sugerida por Bhatt et al. (2021). Para cada ejemplo seleccionado se generan perturbaciones aleatorias que apagan 10% de los puntos de la vista global y local. Luego se calcula la correlacion entre:

- la suma de atribucion XAI en los puntos perturbados;
- la caida de confianza del modelo hacia la clase explicada.

Un valor positivo alto indica que los puntos considerados importantes por el metodo efectivamente producen mayor caida de confianza cuando se perturban. Un valor cercano a cero indica asociacion debil. Un valor negativo sugiere que el mapa puede estar resaltando regiones poco fieles para esa prediccion.

Resumen obtenido:

{faith_corr_table}

Limitaciones: esta metrica no prueba causalidad. Las perturbaciones con valor cero pueden sacar la curva de luz de la distribucion real, y en redes neuronales convolucionales no lineales existen interacciones entre puntos temporales que una correlacion por perturbacion no captura completamente. Por eso se reporta como evidencia cuantitativa complementaria, no como validacion absoluta del metodo XAI. Esto es consistente con la advertencia reciente de que las metricas de fidelidad suelen ser mas confiables en modelos lineales que en modelos no lineales profundos, como senalan Miró-Nicolau et al. (2025).

## Se podran usar estas explicaciones para mejorar el modelo?

Si, pero no conviene usarlas como etiquetas verdaderas. Lo mas seguro es tratarlas como senales auxiliares de diagnostico. Si un falso positivo mira fuera del transito, puede indicar ruido, mala normalizacion, evento secundario o una morfologia confundente. Si un falso negativo tiene atencion dispersa o poca importancia central, puede indicar centrado deficiente, baja senal-ruido o preprocesamiento inadecuado.

Como trabajo futuro, estas explicaciones podrian mejorar el desempeno mediante: auditoria de datos, seleccion de ejemplos dificiles para revision, ajuste de umbrales segun riesgo, entrenamiento de un detector auxiliar de predicciones poco confiables y regularizacion explicable que penalice atencion fuera de zonas fisicamente razonables. Esto siempre debe validarse con metricas en test, porque en una CNN no lineal los mapas XAI no prueban causalidad por si solos.
"""
    text = text.format(faith_corr_table=faith_corr_table)
    (NOTEBOOK_DIR / "README.md").write_text(text, encoding="utf-8")
    (RESULTS_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    prepare_dirs()
    copy_xai_outputs()
    uncertainty_df, outcome_df, dataset_df = compute_uncertainty_tables()
    uncertainty_df.to_csv(TABLES_DIR / "uncertainty_maxlike_mc_dropout_by_sample.csv", index=False)
    outcome_df.to_csv(TABLES_DIR / "uncertainty_maxlike_mc_dropout_by_outcome.csv", index=False)
    dataset_df.to_csv(TABLES_DIR / "uncertainty_maxlike_mc_dropout_by_dataset.csv", index=False)
    for dataset in ("kepler", "tess"):
        plot_maxlike_vs_mc(uncertainty_df, dataset)
        plot_uncertainty_by_outcome(uncertainty_df, dataset)

    faith_sample_df, faith_summary_df = compute_faithfulness_auc_tables()
    faith_sample_df.to_csv(TABLES_DIR / "xai_faithfulness_auc_by_sample.csv", index=False)
    faith_summary_df.to_csv(TABLES_DIR / "xai_faithfulness_auc_summary.csv", index=False)
    plot_faithfulness_auc(faith_summary_df)

    faith_corr_sample_df, faith_corr_summary_df = compute_faithfulness_correlation_tables()
    faith_corr_sample_df.to_csv(TABLES_DIR / "xai_faithfulness_correlation_by_sample.csv", index=False)
    faith_corr_summary_df.to_csv(TABLES_DIR / "xai_faithfulness_correlation_summary.csv", index=False)
    plot_faithfulness_correlation(faith_corr_summary_df)

    quality_df, quality_summary_df = compute_attention_quality_tables()
    quality_df.to_csv(TABLES_DIR / "xai_attention_quality_by_sample.csv", index=False)
    quality_summary_df.to_csv(TABLES_DIR / "xai_attention_quality_summary.csv", index=False)
    plot_attention_quality(quality_df)
    generate_academic_figure_sets(quality_df)

    error_metrics_df = compute_error_detection_metrics(quality_df, uncertainty_df)
    error_metrics_df.to_csv(TABLES_DIR / "xai_error_detection_metrics.csv", index=False)
    plot_error_detection_metrics(error_metrics_df)

    agreement_df, centrality_df = compute_method_agreement_tables()
    agreement_df.to_csv(TABLES_DIR / "xai_method_agreement_summary.csv", index=False)
    centrality_df.to_csv(TABLES_DIR / "xai_method_centrality_summary.csv", index=False)
    plot_method_centrality(centrality_df)

    write_readme(faith_corr_summary_df)
    print(f"Resultados del notebook guardados en: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
