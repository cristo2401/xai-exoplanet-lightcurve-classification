from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TAB = OUT / "tablas"
FIG = OUT / "figuras"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def _run(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return "N/A"


def build_hw_sw_table() -> pd.DataFrame:
    gpu_line = _run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
    cuda_line = _run("nvidia-smi | sed -n '3p' | tr -s ' '")
    os_line = _run("lsb_release -ds 2>/dev/null || cat /etc/os-release | sed -n '1p'")
    cpu_line = _run("lscpu | rg 'Model name' | sed 's/Model name:[ ]*//'")

    py_versions = _run(
        "TF_CPP_MIN_LOG_LEVEL=2 MKL_THREADING_LAYER=GNU python - <<'PY'\n"
        "import sys\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import sklearn\n"
        "import torch\n"
        "import tensorflow as tf\n"
        "print('python=' + sys.version.split()[0])\n"
        "print('torch=' + torch.__version__)\n"
        "print('tensorflow=' + tf.__version__)\n"
        "print('numpy=' + np.__version__)\n"
        "print('pandas=' + pd.__version__)\n"
        "print('sklearn=' + sklearn.__version__)\n"
        "print('torch_cuda=' + str(torch.version.cuda))\n"
        "print('cuda_available=' + str(torch.cuda.is_available()))\n"
        "PY"
    )

    rows = [
        {"Componente": "SO", "Detalle": os_line},
        {"Componente": "Kernel", "Detalle": platform.release()},
        {"Componente": "CPU", "Detalle": cpu_line},
        {"Componente": "CPU hilos lógicos", "Detalle": str(_run("nproc"))},
        {"Componente": "GPU", "Detalle": gpu_line},
        {"Componente": "Info CUDA driver", "Detalle": cuda_line},
        {"Componente": "Entorno Python", "Detalle": py_versions.replace("\n", " | ")},
    ]
    return pd.DataFrame(rows)


def load_core_tables() -> dict[str, pd.DataFrame]:
    baseline = pd.read_csv(ROOT / "resultados_replicacion" / "summary_metrics_with_best_threshold.csv")
    strict = pd.read_csv(ROOT / "resultados_replicacion_strict" / "paper_vs_strict_replication.csv")
    best_effort = pd.read_csv(ROOT / "resultados_replicacion_best_effort" / "summary_best_effort.csv")
    paper_vs_local = pd.read_csv(ROOT / "mejores_resultados" / "mejores_vs_paper.csv")
    return {
        "baseline": baseline,
        "strict": strict,
        "best_effort": best_effort,
        "paper_vs_local": paper_vs_local,
    }


def has_full_sources() -> bool:
    needed = [
        ROOT / "resultados_replicacion" / "summary_metrics_with_best_threshold.csv",
        ROOT / "resultados_replicacion_strict" / "paper_vs_strict_replication.csv",
        ROOT / "resultados_replicacion_best_effort" / "summary_best_effort.csv",
        ROOT / "resultados_tess_official_split" / "tess_dataset_audit.json",
        ROOT / "resultados_kepler_hsearch" / "kepler_dataset_audit.json",
    ]
    return all(p.exists() for p in needed)


def build_evolution_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    baseline = tables["baseline"]
    strict = tables["strict"]
    best_eff = tables["best_effort"]
    pvl = tables["paper_vs_local"]

    for ds in ["Kepler", "TESS"]:
        b = baseline[baseline["Dataset"] == ds].iloc[0]
        rows.append(
            {
                "dataset": ds,
                "stage": "baseline_0.5",
                "objective": "segun configuracion baseline",
                "threshold": 0.5,
                "accuracy": float(b["Acc_at_0_5"]),
                "precision": float(b["Prec_at_0_5"]),
                "recall": float(b["Recall_at_0_5"]),
                "f1": float(b["F1_at_0_5"]),
                "mcc": float(b["MCC_at_0_5"]),
                "ap": float(b["AP"]),
                "source": "resultados_replicacion/summary_metrics_with_best_threshold.csv",
            }
        )

        s = strict[strict["Dataset"] == ds].iloc[0]
        rows.append(
            {
                "dataset": ds,
                "stage": "strict_0.5",
                "objective": "PC vs no-PC (replicación estricta)",
                "threshold": 0.5,
                "accuracy": float(s["Our_Acc@0.5"]),
                "precision": float(s["Our_Prec@0.5"]),
                "recall": float(s["Our_Recall@0.5"]),
                "f1": float(s["Our_F1@0.5"]),
                "mcc": float(s["Our_MCC@0.5"]),
                "ap": float(s["Our_AP"]),
                "source": "resultados_replicacion_strict/paper_vs_strict_replication.csv",
            }
        )

        be = best_eff[best_eff["Dataset"] == ds].iloc[0]
        rows.append(
            {
                "dataset": ds,
                "stage": "best_effort_0.5",
                "objective": "segun barrido best effort",
                "threshold": 0.5,
                "accuracy": float(be["Acc@0.5"]),
                "precision": float(be["Prec@0.5"]),
                "recall": float(be["Recall@0.5"]),
                "f1": float(be["F1@0.5"]),
                "mcc": float(be["MCC@0.5"]),
                "ap": float(be["AP"]),
                "source": "resultados_replicacion_best_effort/summary_best_effort.csv",
            }
        )

        ml = pvl[(pvl["dataset"] == ds) & (pvl["source"] == "mejor_local")].iloc[0]
        rows.append(
            {
                "dataset": ds,
                "stage": "mejor_local_0.5",
                "objective": str(ml["objective"]),
                "threshold": float(ml["threshold"]),
                "accuracy": float(ml["accuracy"]),
                "precision": float(ml["precision"]),
                "recall": float(ml["recall"]),
                "f1": float(ml["f1"]),
                "mcc": float(ml["mcc"]),
                "ap": float(ml["ap"]),
                "source": "mejores_resultados/mejores_vs_paper.csv",
            }
        )

    # Add best-threshold rows from final metric json
    with open(ROOT / "mejores_resultados" / "metricas" / "kepler_mejor_metrics.json", "r", encoding="utf-8") as f:
        km = json.load(f)
    with open(ROOT / "mejores_resultados" / "metricas" / "tess_mejor_metrics.json", "r", encoding="utf-8") as f:
        tm = json.load(f)

    rows.extend(
        [
            {
                "dataset": "Kepler",
                "stage": "mejor_local_best_thr",
                "objective": km["objective"],
                "threshold": float(km["best_thr"]),
                "accuracy": float(km["test_best"]["accuracy"]),
                "precision": float(km["test_best"]["precision"]),
                "recall": float(km["test_best"]["recall"]),
                "f1": float(km["test_best"]["f1"]),
                "mcc": float(km["test_best"]["mcc"]),
                "ap": float(km["test_ap"]),
                "source": "mejores_resultados/metricas/kepler_mejor_metrics.json",
            },
            {
                "dataset": "TESS",
                "stage": "mejor_local_best_thr",
                "objective": tm["objective"],
                "threshold": float(tm["best_thr"]),
                "accuracy": float(tm["test_best"]["accuracy"]),
                "precision": float(tm["test_best"]["precision"]),
                "recall": float(tm["test_best"]["recall"]),
                "f1": float(tm["test_best"]["f1"]),
                "mcc": float(tm["test_best"]["mcc"]),
                "ap": float(tm["test_ap"]),
                "source": "mejores_resultados/metricas/tess_mejor_metrics.json",
            },
        ]
    )

    df = pd.DataFrame(rows)
    return df


def build_audit_table() -> pd.DataFrame:
    with open(ROOT / "resultados_tess_official_split" / "tess_dataset_audit.json", "r", encoding="utf-8") as f:
        ta = json.load(f)
    with open(ROOT / "resultados_kepler_hsearch" / "kepler_dataset_audit.json", "r", encoding="utf-8") as f:
        ka = json.load(f)

    rows = [
        {
            "dataset": "Kepler",
            "paper_total": ka["paper_counts"]["total"],
            "local_total": ka["local_total"]["n"],
            "paper_pos": ka["paper_counts"]["pos"],
            "local_pos": ka["local_total"]["pos"],
            "train_val_overlap": ka["cross_split_overlap"]["train_val"],
            "train_test_overlap": ka["cross_split_overlap"]["train_test"],
            "val_test_overlap": ka["cross_split_overlap"]["val_test"],
        },
        {
            "dataset": "TESS",
            "paper_total": 13207,  # from paper table used in the project
            "local_total": ta["raw_total_canonical"],
            "paper_pos": 391,
            "local_pos": ta["raw_disp_canonical"]["PC"],
            "train_val_overlap": ta["cross_split_overlaps_unique_hash"]["train_val"],
            "train_test_overlap": ta["cross_split_overlaps_unique_hash"]["train_test"],
            "val_test_overlap": ta["cross_split_overlaps_unique_hash"]["val_test"],
        },
    ]
    return pd.DataFrame(rows)


def _save_plot_paper_vs_local():
    source = ROOT / "mejores_resultados" / "mejores_vs_paper.csv"
    if not source.exists():
        source = TAB / "tabla_02_paper_vs_mejor_local.csv"
    df = pd.read_csv(source)
    metrics = ["accuracy", "precision", "recall", "f1", "mcc"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, ds in zip(axes, ["Kepler", "TESS"]):
        sub = df[df["dataset"] == ds]
        paper = sub[sub["source"] == "paper_xie_2025"].iloc[0]
        local = sub[sub["source"] == "mejor_local"].iloc[0]
        x = np.arange(len(metrics))
        w = 0.38
        ax.bar(x - w / 2, [paper[m] for m in metrics], w, label="Paper", color="#2E86AB")
        ax.bar(x + w / 2, [local[m] for m in metrics], w, label="Mejor local", color="#F18F01")
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics], rotation=0)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(ds)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Valor métrico")
    axes[0].legend(loc="lower right")
    fig.suptitle("Paper vs Mejor Resultado Local (threshold=0.5)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_01_paper_vs_local.png", dpi=180)
    plt.close(fig)


def _save_plot_evolution():
    df = pd.read_csv(TAB / "tabla_01_evolucion_preliminar.csv")
    stage_order = [
        "baseline_0.5",
        "strict_0.5",
        "best_effort_0.5",
        "mejor_local_0.5",
        "mejor_local_best_thr",
    ]
    stage_label = {
        "baseline_0.5": "Baseline",
        "strict_0.5": "Strict",
        "best_effort_0.5": "BestEffort",
        "mejor_local_0.5": "Mejor@0.5",
        "mejor_local_best_thr": "Mejor@thr*",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, ds in zip(axes, ["Kepler", "TESS"]):
        sub = df[df["dataset"] == ds].copy()
        sub["order"] = sub["stage"].map({s: i for i, s in enumerate(stage_order)})
        sub = sub.sort_values("order")
        x = np.arange(len(sub))
        w = 0.35
        ax.bar(x - w / 2, sub["f1"].values, width=w, label="F1", color="#3CB371")
        ax.bar(x + w / 2, sub["mcc"].values, width=w, label="MCC", color="#4169E1")
        ax.set_xticks(x)
        ax.set_xticklabels([stage_label[s] for s in sub["stage"]], rotation=20)
        ax.set_ylim(0, 1.05)
        ax.set_title(ds)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[0].legend(loc="lower right")
    fig.suptitle("Evolución de Resultados Preliminares")
    fig.tight_layout()
    fig.savefig(FIG / "fig_02_evolucion_f1_mcc.png", dpi=180)
    plt.close(fig)


def _save_plot_top_hsearch():
    k_src = ROOT / "resultados_kepler_hsearch" / "kepler_hsearch_summary.csv"
    t_src = ROOT / "resultados_tess_triage_hsearch" / "hsearch_summary.csv"
    if not k_src.exists():
        k_src = TAB / "fuente_kepler_hsearch_summary.csv"
    if not t_src.exists():
        t_src = TAB / "fuente_tess_hsearch_summary.csv"
    k = pd.read_csv(k_src).head(8)
    t = pd.read_csv(t_src).head(8)

    def _plot(df: pd.DataFrame, title: str, outname: str):
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        y = np.arange(len(df))
        ax.barh(y, df["f1@best"], color="#5DA5DA", label="F1@best")
        ax.scatter(df["mcc@best"], y, color="#D62728", s=55, label="MCC@best")
        ax.set_yticks(y)
        ax.set_yticklabels(df["name"])
        ax.invert_yaxis()
        ax.set_xlim(0.75, 1.0)
        ax.set_xlabel("Score")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(FIG / outname, dpi=180)
        plt.close(fig)

    _plot(k, "Top configuraciones Kepler (búsqueda hiperparámetros)", "fig_03_top_configs_kepler.png")
    _plot(t, "Top configuraciones TESS (búsqueda hiperparámetros)", "fig_04_top_configs_tess.png")


def _save_plot_data_audit():
    ad = pd.read_csv(TAB / "tabla_03_auditoria_datos.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, ds in zip(axes, ["Kepler", "TESS"]):
        r = ad[ad["dataset"] == ds].iloc[0]
        x = np.arange(2)
        ax.bar(x - 0.15, [r["paper_total"], r["local_total"]], width=0.3, label="Total", color="#7AA974")
        ax.bar(x + 0.15, [r["paper_pos"], r["local_pos"]], width=0.3, label="Positivos", color="#C44E52")
        ax.set_xticks(x)
        ax.set_xticklabels(["Paper", "Local"])
        ax.set_title(ds)
        ax.grid(axis="y", alpha=0.25)
        ov = f"overlap train-val/test: {int(r['train_val_overlap'])}/{int(r['train_test_overlap'])}, val-test: {int(r['val_test_overlap'])}"
        ax.text(0.02, 0.98, ov, transform=ax.transAxes, va="top", fontsize=8)
    axes[0].set_ylabel("N° muestras")
    axes[1].legend(loc="upper right")
    fig.suptitle("Auditoría de datos: Paper vs datos locales")
    fig.tight_layout()
    fig.savefig(FIG / "fig_05_auditoria_datos.png", dpi=180)
    plt.close(fig)


def copy_reference_plots():
    copies = [
        (ROOT / "resultados_replicacion" / "kepler_pr_curve.png", FIG / "ref_kepler_pr_curve_baseline.png"),
        (ROOT / "resultados_replicacion" / "tess_pr_curve.png", FIG / "ref_tess_pr_curve_baseline.png"),
        (ROOT / "resultados_replicacion_strict" / "kepler_strict_pr_curve.png", FIG / "ref_kepler_pr_curve_strict.png"),
        (ROOT / "resultados_replicacion_strict" / "tess_strict_pr_curve.png", FIG / "ref_tess_pr_curve_strict.png"),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copy2(src, dst)


def main():
    hw = build_hw_sw_table()
    hw.to_csv(TAB / "tabla_04_hw_sw.csv", index=False)

    if has_full_sources():
        tables = load_core_tables()
        pvl = tables["paper_vs_local"].copy()
        pvl.to_csv(TAB / "tabla_02_paper_vs_mejor_local.csv", index=False)

        evo = build_evolution_table(tables)
        evo.to_csv(TAB / "tabla_01_evolucion_preliminar.csv", index=False)

        audit = build_audit_table()
        audit.to_csv(TAB / "tabla_03_auditoria_datos.csv", index=False)

    _save_plot_paper_vs_local()
    _save_plot_evolution()
    _save_plot_top_hsearch()
    _save_plot_data_audit()
    copy_reference_plots()

    print("OK: tablas y figuras generadas en", OUT)


if __name__ == "__main__":
    main()
