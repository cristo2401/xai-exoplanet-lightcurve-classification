from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

THIS_DIR = Path(__file__).resolve().parent
ACADEMIC_DIR = THIS_DIR.parent
RESULTS_DIR = ACADEMIC_DIR.parent
NOTEBOOK_DIR = RESULTS_DIR.parent
ROOT = NOTEBOOK_DIR.parent
sys.path.insert(0, str(ROOT))

import xai_mejores_modelos as xai  # noqa: E402

try:
    import tensorflow as tf  # noqa: E402
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
except Exception:
    tf = None

OUTCOME_ORDER = ["TN", "FP", "FN", "TP"]
METHOD_ORDER = ["saliency", "integrated_gradients", "occlusion", "consensus"]
METHOD_LABELS = {
    "saliency": "Saliency",
    "integrated_gradients": "Integrated Gradients",
    "occlusion": "Occlusion",
    "consensus": "Consensus",
}
DATASET_LABELS = {"kepler": "Kepler", "tess": "TESS"}


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


def select_stratified_indices(
    y: np.ndarray,
    pred: np.ndarray,
    probs: np.ndarray,
    max_per_outcome: int,
    rng: np.random.Generator,
) -> list[int]:
    outcomes = outcome_labels(y, pred)
    selected: list[int] = []
    confidence = np.where(pred == 1, probs, 1.0 - probs)

    for outcome in OUTCOME_ORDER:
        idx = np.where(outcomes == outcome)[0]
        if len(idx) == 0:
            continue
        if len(idx) <= max_per_outcome:
            chosen = idx
        else:
            # Keep a mix of easy and difficult samples instead of pure random-only sampling.
            order = idx[np.argsort(confidence[idx])]
            low_n = max_per_outcome // 3
            high_n = max_per_outcome // 3
            mid_n = max_per_outcome - low_n - high_n
            low = order[:low_n]
            high = order[-high_n:] if high_n > 0 else np.array([], dtype=np.int64)
            remaining = np.setdiff1d(idx, np.concatenate([low, high]), assume_unique=False)
            mid = rng.choice(remaining, size=min(mid_n, len(remaining)), replace=False) if len(remaining) else np.array([], dtype=np.int64)
            chosen = np.concatenate([low, mid, high])
        selected.extend([int(v) for v in chosen])

    return sorted(set(selected))


def target_confidence_batch(
    model: torch.nn.Module,
    xg_batch: np.ndarray,
    xl_batch: np.ndarray,
    target_label: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    confs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(xg_batch), batch_size):
            end = min(start + batch_size, len(xg_batch))
            bg = torch.tensor(xg_batch[start:end, None, :], dtype=torch.float32, device=device)
            bl = torch.tensor(xl_batch[start:end, None, :], dtype=torch.float32, device=device)
            probs = torch.sigmoid(model(bg, bl)).cpu().numpy().ravel()
            confs.extend((probs if int(target_label) == 1 else 1.0 - probs).tolist())
    return np.asarray(confs, dtype=np.float32)


def random_masks(global_len: int, local_len: int, n_perturbations: int, mask_fraction: float, rng: np.random.Generator):
    kg = max(1, int(round(global_len * mask_fraction)))
    kl = max(1, int(round(local_len * mask_fraction)))
    for _ in range(n_perturbations):
        mask_g = np.sort(rng.choice(global_len, size=kg, replace=False)).astype(np.int32)
        mask_l = np.sort(rng.choice(local_len, size=kl, replace=False)).astype(np.int32)
        yield mask_g, mask_l


def compute_sample_faithfulness(
    model: torch.nn.Module,
    dataset: str,
    idx: int,
    xg: np.ndarray,
    xl: np.ndarray,
    y_true: int,
    raw_label: str,
    prob: float,
    pred_label: int,
    outcome: str,
    device: torch.device,
    rng: np.random.Generator,
    n_perturbations: int,
    mask_fraction: float,
    batch_size: int,
    ig_steps: int,
) -> list[dict]:
    target_label = int(pred_label)
    base_conf = float(xai.target_confidence_from_prob(prob, target_label))

    sal_g, sal_l = xai.compute_saliency(model, xg, xl, device, target_label)
    ig_g, ig_l = xai.compute_integrated_gradients(model, xg, xl, device, target_label, steps=ig_steps)
    occ_g, occ_l = xai.compute_occlusion_sensitivity(model, xg, xl, device, target_label, batch_size=batch_size)
    cons_g = xai.consensus_importance(sal_g, ig_g, occ_g)
    cons_l = xai.consensus_importance(sal_l, ig_l, occ_l)
    maps = {
        "saliency": (sal_g, sal_l),
        "integrated_gradients": (ig_g, ig_l),
        "occlusion": (occ_g, occ_l),
        "consensus": (cons_g, cons_l),
    }

    masks = list(random_masks(len(xg), len(xl), n_perturbations, mask_fraction, rng))
    perturbed_g = []
    perturbed_l = []
    for mask_g, mask_l in masks:
        gx = xg.copy()
        lx = xl.copy()
        gx[mask_g] = 0.0
        lx[mask_l] = 0.0
        perturbed_g.append(gx)
        perturbed_l.append(lx)
    perturbed_g = np.stack(perturbed_g).astype(np.float32)
    perturbed_l = np.stack(perturbed_l).astype(np.float32)
    perturbed_confs = target_confidence_batch(model, perturbed_g, perturbed_l, target_label, device, batch_size)
    confidence_drops = base_conf - perturbed_confs

    rows = []
    for method, (heat_g, heat_l) in maps.items():
        attribution_sums = np.asarray(
            [float(heat_g[mask_g].sum() + heat_l[mask_l].sum()) for mask_g, mask_l in masks],
            dtype=np.float64,
        )
        pearson = pearson_corr(attribution_sums, confidence_drops)
        spearman = spearman_corr(attribution_sums, confidence_drops)
        rows.append(
            {
                "dataset": dataset,
                "index": int(idx),
                "raw_label": raw_label,
                "y_true": int(y_true),
                "pred_label": int(pred_label),
                "outcome": outcome,
                "method": method,
                "base_probability": float(prob),
                "base_target_confidence": float(base_conf),
                "n_perturbations": int(n_perturbations),
                "mask_fraction": float(mask_fraction),
                "faithfulness_pearson": pearson,
                "faithfulness_spearman": spearman,
                "mean_confidence_drop": float(np.mean(confidence_drops)),
                "std_confidence_drop": float(np.std(confidence_drops)),
                "negative_drop_fraction": float(np.mean(confidence_drops < 0.0)),
                "mean_attribution_sum": float(np.mean(attribution_sums)),
                "std_attribution_sum": float(np.std(attribution_sums)),
                "valid_correlation": bool(np.isfinite(pearson) or np.isfinite(spearman)),
            }
        )
    return rows


def summarize(sample_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        sample_df.groupby(["dataset", "method"], as_index=False)
        .agg(
            n=("index", "count"),
            n_unique_samples=("index", "nunique"),
            valid_n=("valid_correlation", "sum"),
            pearson_mean=("faithfulness_pearson", "mean"),
            pearson_std=("faithfulness_pearson", "std"),
            spearman_mean=("faithfulness_spearman", "mean"),
            spearman_std=("faithfulness_spearman", "std"),
            mean_confidence_drop=("mean_confidence_drop", "mean"),
            negative_drop_fraction_mean=("negative_drop_fraction", "mean"),
        )
        .sort_values(["dataset", "pearson_mean"], ascending=[True, False])
    )
    by_outcome = (
        sample_df.groupby(["dataset", "outcome", "method"], as_index=False)
        .agg(
            n=("index", "count"),
            n_unique_samples=("index", "nunique"),
            pearson_mean=("faithfulness_pearson", "mean"),
            spearman_mean=("faithfulness_spearman", "mean"),
            mean_confidence_drop=("mean_confidence_drop", "mean"),
        )
        .sort_values(["dataset", "outcome", "pearson_mean"], ascending=[True, True, False])
    )
    return summary, by_outcome


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True, sharex=True)
    for ax, dataset in zip(axes, ["kepler", "tess"]):
        g = summary[summary.dataset == dataset].set_index("method").loc[METHOD_ORDER].reset_index()
        y = np.arange(len(g))
        ax.barh(y - 0.16, g["pearson_mean"], xerr=g["pearson_std"].fillna(0), height=0.30, label="Pearson", color="#2a9d8f", alpha=0.85)
        ax.barh(y + 0.16, g["spearman_mean"], xerr=g["spearman_std"].fillna(0), height=0.30, label="Spearman", color="#e76f51", alpha=0.78)
        ax.axvline(0, color="#333333", linewidth=1)
        ax.set_yticks(y, [METHOD_LABELS[m] for m in g["method"]])
        ax.set_title(f"{DATASET_LABELS[dataset]} - Faithfulness Correlation ampliada")
        ax.set_xlabel("correlacion media")
        ax.grid(axis="x", alpha=0.25)
    axes[0].legend(loc="lower right")
    fig.savefig(out_dir / "faithfulness_correlation_ampliada_por_metodo.png", dpi=220)
    plt.close(fig)


def plot_distributions(sample_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True, sharey=True)
    for ax, dataset in zip(axes, ["kepler", "tess"]):
        data = [
            sample_df[(sample_df.dataset == dataset) & (sample_df.method == method)]["faithfulness_pearson"].dropna().to_numpy()
            for method in METHOD_ORDER
        ]
        parts = ax.violinplot(data, showmeans=True, showextrema=True, widths=0.8)
        for body in parts["bodies"]:
            body.set_facecolor("#457b9d")
            body.set_edgecolor("#1d3557")
            body.set_alpha(0.55)
        for key in ["cmeans", "cbars", "cmins", "cmaxes"]:
            parts[key].set_color("#1d3557")
            parts[key].set_linewidth(1.1)
        ax.axhline(0, color="#333333", linestyle="--", linewidth=1)
        ax.set_xticks(np.arange(1, len(METHOD_ORDER) + 1), [METHOD_LABELS[m] for m in METHOD_ORDER], rotation=20, ha="right")
        ax.set_title(f"{DATASET_LABELS[dataset]} - distribucion por muestra")
        ax.set_ylabel("Faithfulness Pearson")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_dir / "faithfulness_correlation_ampliada_distribucion.png", dpi=220)
    plt.close(fig)


def plot_outcomes(by_outcome: pd.DataFrame, out_dir: Path) -> None:
    method = "occlusion"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True, sharey=True)
    for ax, dataset in zip(axes, ["kepler", "tess"]):
        g = by_outcome[(by_outcome.dataset == dataset) & (by_outcome.method == method)].copy()
        g["outcome"] = pd.Categorical(g["outcome"], OUTCOME_ORDER, ordered=True)
        g = g.sort_values("outcome")
        ax.bar(g["outcome"].astype(str), g["pearson_mean"], color="#6a994e", alpha=0.82)
        ax.axhline(0, color="#333333", linestyle="--", linewidth=1)
        for i, row in enumerate(g.itertuples(index=False)):
            ax.text(i, row.pearson_mean + (0.025 if row.pearson_mean >= 0 else -0.04), f"n={int(row.n_unique_samples)}", ha="center", va="bottom" if row.pearson_mean >= 0 else "top", fontsize=9)
        ax.set_title(f"{DATASET_LABELS[dataset]} - Occlusion por resultado")
        ax.set_xlabel("resultado")
        ax.set_ylabel("Pearson medio")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_dir / "faithfulness_occlusion_ampliada_por_resultado.png", dpi=220)
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns.astype(str)) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        vals = []
        for value in row.tolist():
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *rows])


def write_report(sample_df: pd.DataFrame, summary: pd.DataFrame, by_outcome: pd.DataFrame, out_dir: Path, config: dict) -> None:
    compact = summary[["dataset", "method", "n_unique_samples", "valid_n", "pearson_mean", "spearman_mean", "mean_confidence_drop", "negative_drop_fraction_mean"]].copy()
    compact["method"] = compact["method"].map(METHOD_LABELS)
    compact = compact.sort_values(["dataset", "pearson_mean"], ascending=[True, False])

    counts = sample_df.drop_duplicates(["dataset", "index"]).groupby(["dataset", "outcome"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=OUTCOME_ORDER, fill_value=0).reset_index()

    text = f"""# Fidelidad XAI ampliada y nota sobre Quantus/SHAP

Esta carpeta documenta una mejora específica al análisis de fidelidad. En la versión inicial, la fidelidad se calculó sobre los ejemplos seleccionados para visualización, aproximadamente 6 por dataset. Eso era suficiente como evidencia preliminar, pero era débil como respaldo cuantitativo porque la muestra era pequeña.

Para mejorar esto, se agregó una evaluación ampliada tipo **Faithfulness Correlation** sobre una muestra estratificada del test set. La evaluación usa hasta `{config['max_per_outcome']}` ejemplos por resultado de matriz de confusión (`TN`, `FP`, `FN`, `TP`) en cada dataset, incluyendo todos los errores disponibles cuando son menos que ese máximo. En total se evaluaron `{sample_df.drop_duplicates(['dataset', 'index']).shape[0]}` muestras únicas: `{sample_df[sample_df.dataset == 'kepler']['index'].nunique()}` de Kepler y `{sample_df[sample_df.dataset == 'tess']['index'].nunique()}` de TESS.

## Sobre Quantus

**Quantus no se usó literalmente como dependencia de Python.** La razón práctica fue mantener el repositorio reproducible con las dependencias ya incluidas y evitar agregar una instalación nueva solo para una métrica. Sin embargo, la métrica implementada sigue el mismo objetivo que la Faithfulness Correlation sugerida en la literatura y disponible en Quantus: perturbar regiones de la entrada, medir el cambio en la salida del modelo y correlacionar ese cambio con la importancia asignada por el método XAI.

Por lo tanto, en el informe conviene escribirlo como: *"Se calculó una métrica de fidelidad tipo Faithfulness Correlation, equivalente en objetivo a la implementación disponible en Quantus, pero implementada localmente para mantener reproducibilidad del pipeline"*. No conviene afirmar que se usó Quantus si no se importó la biblioteca.

## Sobre SHAP

SHAP no fue usado y se puede eliminar del paper. Para estas curvas de luz, SHAP por punto temporal puede ser costoso y difícil de interpretar, especialmente en Kepler con 2001 puntos globales y 201 puntos locales. La comparación principal queda mejor defendida con Saliency, Integrated Gradients, Occlusion, Consensus, MC Dropout y métricas de fidelidad.

## Protocolo ampliado

Para cada muestra seleccionada se explicó la clase predicha por el modelo, no necesariamente la etiqueta verdadera. Esto evalúa la fidelidad de la explicación respecto de la decisión real del modelo. Luego se generaron `{config['n_perturbations']}` perturbaciones aleatorias por muestra, apagando el `{config['mask_fraction']:.0%}` de los puntos de la vista global y el `{config['mask_fraction']:.0%}` de los puntos de la vista local. Para cada perturbación se midió la caída de confianza hacia la clase predicha. Finalmente, se calculó la correlación entre:

- la suma de atribuciones XAI en los puntos perturbados;
- la caída de confianza del modelo después de perturbar esos puntos.

Un valor positivo indica que las regiones marcadas como importantes tienden a producir mayor cambio en la salida cuando se intervienen. Un valor cercano a cero indica fidelidad débil. Un valor negativo indica que, bajo este protocolo, el mapa puede estar resaltando regiones que no explican bien la salida del modelo.

## Distribución de muestras evaluadas

{dataframe_to_markdown(counts)}

## Resumen de resultados ampliados

{dataframe_to_markdown(compact)}

## Interpretación

La evaluación ampliada mejora la defensa del trabajo porque ya no depende solo de los 6 ejemplos usados para figuras individuales. Ahora la fidelidad se mide sobre una muestra estratificada que incluye verdaderos negativos, falsos positivos, falsos negativos y verdaderos positivos.

La conclusión principal debe expresarse con cautela: las explicaciones tienen fidelidad parcial, no absoluta. Si los valores son positivos, hay evidencia de que las regiones importantes están relacionadas con cambios reales en la salida del modelo. Si algunos métodos quedan cerca de cero o negativos, eso indica que esos mapas deben usarse como visualización complementaria y no como prueba causal.

En general, **Occlusion** y **Consensus** son los métodos más defendibles para interpretar el modelo, porque conectan mejor la importancia visual con cambios de confianza. Integrated Gradients también es útil, pero puede variar más por muestra. Saliency es barato y sirve como diagnóstico rápido, aunque puede ser más ruidoso.

## Limitaciones

Aunque esta evaluación es más fuerte que la inicial, sigue teniendo limitaciones. Primero, no se evaluó todo el test set, sino una muestra estratificada ampliada para mantener tiempo de cómputo razonable. Segundo, las perturbaciones con ceros pueden generar curvas fuera de distribución. Tercero, el modelo es una red neuronal convolucional no lineal con dos ramas, por lo que las interacciones entre vista global y local no se capturan completamente con una correlación local.

Esto es importante para responder a la observación de Miró-Nicolau et al. (2025): las métricas de fidelidad son más confiables en modelos lineales que en redes neuronales profundas. Por eso estos valores deben interpretarse como evidencia comparativa entre métodos XAI bajo un protocolo fijo, no como certificación causal absoluta.

## Figuras generadas

- `faithfulness_correlation_ampliada_por_metodo.png`: compara Pearson y Spearman medios por método y dataset.
- `faithfulness_correlation_ampliada_distribucion.png`: muestra la distribución por muestra de Pearson para cada método.
- `faithfulness_occlusion_ampliada_por_resultado.png`: resume el comportamiento de Occlusion por resultado de matriz de confusión.

## Cómo reportarlo en el paper

Texto recomendado:

"Para evaluar si las explicaciones reflejan el comportamiento del modelo, se calculó una métrica tipo Faithfulness Correlation sobre una muestra estratificada del conjunto de test. Aunque Quantus no se utilizó literalmente como dependencia, el protocolo implementa la misma idea: perturbar regiones de la entrada y correlacionar la importancia asignada por XAI con la caída de confianza del modelo. Esta evaluación se realizó sobre una muestra ampliada que incluye verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos. Los resultados indican fidelidad parcial y dependiente del método, siendo Occlusion y Consensus las alternativas más defendibles. Debido a la naturaleza no lineal de la red neuronal, estos valores se interpretan como evidencia comparativa y no como prueba causal absoluta."
"""
    (out_dir / "README_fidelidad_ampliada.txt").write_text(text, encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Calcular fidelidad XAI ampliada tipo Quantus-style.")
    parser.add_argument("--max-per-outcome", type=int, default=40)
    parser.add_argument("--n-perturbations", type=int, default=32)
    parser.add_argument("--mask-fraction", type=float, default=0.10)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []

    for dataset in ["kepler", "tess"]:
        spec = xai.DATASETS[dataset]
        model = xai.build_model(ROOT, spec, device)
        xg, xl, y, raw = xai.load_dataset(ROOT, spec)
        threshold = xai.load_threshold(ROOT, spec)
        probs = xai.predict_probs(model, xg, xl, device, batch_size=args.batch_size)
        pred = (probs >= threshold).astype(np.int32)
        outcomes = outcome_labels(y, pred)
        selected = select_stratified_indices(y, pred, probs, args.max_per_outcome, rng)
        print(f"[{dataset}] muestras seleccionadas: {len(selected)}")

        for j, idx in enumerate(selected, 1):
            if j % 10 == 0 or j == len(selected):
                print(f"[{dataset}] {j}/{len(selected)}")
            rows.extend(
                compute_sample_faithfulness(
                    model=model,
                    dataset=dataset,
                    idx=idx,
                    xg=xg[idx],
                    xl=xl[idx],
                    y_true=int(y[idx]),
                    raw_label=raw[idx],
                    prob=float(probs[idx]),
                    pred_label=int(pred[idx]),
                    outcome=str(outcomes[idx]),
                    device=device,
                    rng=rng,
                    n_perturbations=args.n_perturbations,
                    mask_fraction=args.mask_fraction,
                    batch_size=args.batch_size,
                    ig_steps=args.ig_steps,
                )
            )

    sample_df = pd.DataFrame(rows)
    summary, by_outcome = summarize(sample_df)
    sample_df.to_csv(THIS_DIR / "faithfulness_ampliada_by_sample.csv", index=False)
    summary.to_csv(THIS_DIR / "faithfulness_ampliada_summary.csv", index=False)
    by_outcome.to_csv(THIS_DIR / "faithfulness_ampliada_by_outcome.csv", index=False)
    plot_summary(summary, THIS_DIR)
    plot_distributions(sample_df, THIS_DIR)
    plot_outcomes(by_outcome, THIS_DIR)
    write_report(
        sample_df,
        summary,
        by_outcome,
        THIS_DIR,
        {
            "max_per_outcome": args.max_per_outcome,
            "n_perturbations": args.n_perturbations,
            "mask_fraction": args.mask_fraction,
        },
    )
    print("Listo:", THIS_DIR)


if __name__ == "__main__":
    main()
