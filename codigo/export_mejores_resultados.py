import glob
import hashlib
import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import average_precision_score, matthews_corrcoef, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "mejores_resultados"
MODELS_DIR = OUT_DIR / "modelos"
METRICS_DIR = OUT_DIR / "metricas"
for d in (OUT_DIR, MODELS_DIR, METRICS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_views(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return x.astype(np.float32)
    if mode == "zscore":
        mu = np.mean(x, axis=1, keepdims=True)
        sd = np.std(x, axis=1, keepdims=True) + 1e-6
        return ((x - mu) / sd).astype(np.float32)
    if mode == "robust":
        med = np.median(x, axis=1, keepdims=True)
        sd = np.std(x, axis=1, keepdims=True) + 1e-6
        z = (x - med) / sd
        return np.clip(z, -5.0, 5.0).astype(np.float32)
    raise ValueError(mode)


class CurveDS(Dataset):
    def __init__(self, xg, xl, y, flip_prob=0.0, noise_std=0.0):
        self.xg = np.expand_dims(xg.astype(np.float32), axis=1)
        self.xl = np.expand_dims(xl.astype(np.float32), axis=1)
        self.y = y.astype(np.float32)
        self.flip_prob = float(flip_prob)
        self.noise_std = float(noise_std)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        xg = self.xg[idx].copy()
        xl = self.xl[idx].copy()

        if self.flip_prob > 0.0 and np.random.rand() < self.flip_prob:
            xg = np.flip(xg, axis=1).copy()
            xl = np.flip(xl, axis=1).copy()

        if self.noise_std > 0.0:
            xg += np.random.normal(0.0, self.noise_std, size=xg.shape).astype(np.float32)
            xl += np.random.normal(0.0, self.noise_std, size=xl.shape).astype(np.float32)

        return torch.from_numpy(xg), torch.from_numpy(xl), torch.tensor([self.y[idx]], dtype=torch.float32)


class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResNetUnit(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.d1 = nn.Linear(n, n)
        self.d2 = nn.Linear(n, n)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        y = self.act(self.d1(x))
        y = self.d2(y)
        return self.act(y + x)


class Net(nn.Module):
    def __init__(self, gl: int, ll: int, drop: float):
        super().__init__()
        self.gb = nn.Sequential(
            nn.Conv1d(1, 16, 5, 1, 2), nn.ReLU(),
            nn.Conv1d(16, 16, 5, 1, 2), nn.ReLU(), nn.MaxPool1d(5, 2),
            nn.Conv1d(16, 32, 5, 1, 2), nn.ReLU(),
            nn.Conv1d(32, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool1d(5, 2),
            nn.Conv1d(32, 64, 5, 1, 2), nn.ReLU(),
            nn.Conv1d(64, 64, 5, 1, 2), nn.ReLU(), nn.MaxPool1d(5, 2),
            nn.Conv1d(64, 128, 5, 1, 2), nn.ReLU(),
            nn.Conv1d(128, 128, 5, 1, 2), nn.ReLU(), nn.MaxPool1d(5, 2),
            nn.Conv1d(128, 256, 5, 1, 2), nn.ReLU(),
            nn.Conv1d(256, 256, 5, 1, 2), nn.ReLU(), nn.MaxPool1d(5, 2),
        )
        self.lb = nn.Sequential(
            nn.Conv1d(1, 16, 5, 1, 2), nn.ReLU(), SEBlock1D(16),
            nn.Conv1d(16, 16, 5, 1, 2), nn.ReLU(), SEBlock1D(16), nn.MaxPool1d(5, 2),
            nn.Conv1d(16, 32, 5, 1, 2), nn.ReLU(), SEBlock1D(32),
            nn.Conv1d(32, 32, 5, 1, 2), nn.ReLU(), SEBlock1D(32), nn.MaxPool1d(5, 2),
        )
        with torch.no_grad():
            f = self.gb(torch.zeros(1, 1, gl)).numel() + self.lb(torch.zeros(1, 1, ll)).numel()
        self.fc = nn.Linear(f, 512)
        self.drop = nn.Dropout(drop)
        self.r = nn.Sequential(*[ResNetUnit(512) for _ in range(6)])
        self.o = nn.Linear(512, 1)

    def forward(self, xg, xl):
        g = self.gb(xg).reshape(xg.size(0), -1)
        l = self.lb(xl).reshape(xl.size(0), -1)
        z = torch.cat((g, l), 1)
        z = F.leaky_relu(self.fc(z), 0.1)
        z = self.drop(z)
        z = self.r(z)
        return self.o(z)


def eval_probs(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xg, xl, y in loader:
            logits = model(xg.to(device), xl.to(device))
            ps.append(torch.sigmoid(logits).cpu().numpy().ravel())
            ys.append(y.numpy().ravel())
    return np.concatenate(ys).astype(np.int32), np.concatenate(ps)


def metrics(y, p, t=0.5):
    pred = (p >= t).astype(np.int32)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    mcc = matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0
    return {
        "accuracy": float((pred == y).mean()),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "mcc": float(mcc),
        "pred_pos_rate": float(pred.mean()),
    }


def train_model(
    xg_tr,
    xl_tr,
    y_tr,
    xg_va,
    xl_va,
    y_va,
    xg_te,
    xl_te,
    y_te,
    *,
    gl,
    ll,
    drop,
    lr,
    weight_decay,
    epochs,
    patience,
    flip_prob,
    noise_std,
    use_pos_weight,
    seed,
):
    set_seed(seed)

    tr_loader = DataLoader(CurveDS(xg_tr, xl_tr, y_tr, flip_prob=flip_prob, noise_std=noise_std), batch_size=128, shuffle=True, num_workers=0)
    va_loader = DataLoader(CurveDS(xg_va, xl_va, y_va), batch_size=256, shuffle=False, num_workers=0)
    te_loader = DataLoader(CurveDS(xg_te, xl_te, y_te), batch_size=256, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(gl=gl, ll=ll, drop=drop).to(device)

    if use_pos_weight:
        pos = max(int((y_tr == 1).sum()), 1)
        neg = max(int((y_tr == 0).sum()), 1)
        pw = torch.tensor([neg / pos], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        criterion = nn.BCEWithLogitsLoss()

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = optim.lr_scheduler.StepLR(opt, step_size=max(10, epochs // 2), gamma=0.5)

    best_state = None
    best_score = (-1.0, -1.0)
    best_thr = 0.5
    bad = 0

    for _ in range(epochs):
        model.train()
        for xg, xl, y in tr_loader:
            xg, xl, y = xg.to(device), xl.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xg, xl)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
        sch.step()

        yv, pv = eval_probs(model, va_loader, device)
        f05 = metrics(yv, pv, 0.5)["f1"]
        ths = np.arange(0.1, 0.91, 0.05)
        t_best, f_best = max([(float(t), metrics(yv, pv, float(t))["f1"]) for t in ths], key=lambda z: z[1])
        score = (f05, f_best)

        if score > best_score:
            best_score = score
            best_thr = t_best
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)

    yt, pt = eval_probs(model, te_loader, device)
    m05 = metrics(yt, pt, 0.5)
    mb = metrics(yt, pt, best_thr)
    ap = float(average_precision_score(yt, pt))

    return model, {
        "best_val_f1_0_5": float(best_score[0]),
        "best_val_f1_any": float(best_score[1]),
        "best_thr": float(best_thr),
        "test_0_5": m05,
        "test_best": mb,
        "test_ap": ap,
        "sizes": {
            "train": int(len(y_tr)),
            "val": int(len(y_va)),
            "test": int(len(y_te)),
            "train_pos": int((y_tr == 1).sum()),
            "val_pos": int((y_va == 1).sum()),
            "test_pos": int((y_te == 1).sum()),
        },
    }


def load_kepler_splits():
    with h5py.File(BASE_DIR / "datos_procesados_h5/kepler_dataset.h5", "r") as hf:
        xg_tr = hf["train"]["global_view"][:].astype(np.float32)
        xl_tr = hf["train"]["local_view"][:].astype(np.float32)
        y_tr = hf["train"]["labels"][:].astype(np.int32)

        xg_va = hf["val"]["global_view"][:].astype(np.float32)
        xl_va = hf["val"]["local_view"][:].astype(np.float32)
        y_va = hf["val"]["labels"][:].astype(np.int32)

        xg_te = hf["test"]["global_view"][:].astype(np.float32)
        xl_te = hf["test"]["local_view"][:].astype(np.float32)
        y_te = hf["test"]["labels"][:].astype(np.int32)
    return (xg_tr, xl_tr, y_tr, xg_va, xl_va, y_va, xg_te, xl_te, y_te)


def load_tess_triage_splits():
    base = BASE_DIR / "tfrecords_TESS"
    train_files = sorted(glob.glob(str(base / "train-*")))
    val_files = sorted(glob.glob(str(base / "val-*")))
    test_files = sorted(glob.glob(str(base / "test-*")))

    def rows(files):
        out = []
        for rec in tf.data.TFRecordDataset(files):
            ex = tf.train.Example()
            ex.ParseFromString(rec.numpy())
            f = ex.features.feature
            disp = f["Disposition"].bytes_list.value[0].decode()
            g = np.array(f["global_view"].float_list.value, dtype=np.float32)
            l = np.array(f["local_view"].float_list.value, dtype=np.float32)
            h = hashlib.sha1(np.concatenate([g, l]).tobytes()).hexdigest()
            out.append((h, disp, g, l))

        seen = set()
        dedup = []
        for h, disp, g, l in out:
            if h in seen:
                continue
            seen.add(h)
            dedup.append((disp, g, l))
        return dedup

    tr = rows(train_files)
    va = rows(val_files)
    te = rows(test_files)

    def to_arrays(rs):
        xg = np.stack([r[1] for r in rs], axis=0).astype(np.float32)
        xl = np.stack([r[2] for r in rs], axis=0).astype(np.float32)
        # objetivo triage: positivo = PC o EB
        y = np.array([1 if r[0] in {"PC", "EB"} else 0 for r in rs], dtype=np.int32)
        return xg, xl, y

    return (*to_arrays(tr), *to_arrays(va), *to_arrays(te))


def build_readme(kepler_metrics, tess_metrics):
    text = f"""# mejores_resultados

Esta carpeta contiene los mejores modelos y métricas observadas en esta réplica para **Kepler** y **TESS**.

## Contenido
- `modelos/kepler_mejor_modelo.pth`: modelo campeón Kepler.
- `modelos/tess_mejor_modelo.pth`: modelo campeón TESS.
- `metricas/kepler_mejor_metrics.json`: métricas y configuración del mejor Kepler.
- `metricas/tess_mejor_metrics.json`: métricas y configuración del mejor TESS.
- `mejores_vs_paper.csv`: comparación Paper vs mejor local para ambos datasets.

## Kepler (mejor local)
- Arquitectura: SE-CNN-RINet (misma familia usada en los scripts del proyecto).
- Datos: `datos_procesados_h5/kepler_dataset.h5` (`train/val/test` existentes).
- Configuración:
  - `preprocess = zscore`
  - `loss = BCE`
  - `lr = 2e-4`
  - `dropout = 0.1`
  - `flip_prob = 0.3`
  - `noise_std = 0.005`
  - `seed = 29`
- Resultado en test @0.5:
  - `accuracy = {kepler_metrics['test_0_5']['accuracy']:.6f}`
  - `precision = {kepler_metrics['test_0_5']['precision']:.6f}`
  - `recall = {kepler_metrics['test_0_5']['recall']:.6f}`
  - `f1 = {kepler_metrics['test_0_5']['f1']:.6f}`
  - `mcc = {kepler_metrics['test_0_5']['mcc']:.6f}`
  - `ap = {kepler_metrics['test_ap']:.6f}`

## TESS (mejor local)
- Arquitectura: SE-CNN-RINet (misma familia usada en los scripts del proyecto).
- Datos: `tfrecords_TESS` con splits `train/val/test` (deduplicación exacta por hash dentro de cada split).
- Configuración:
  - `preprocess = robust`
  - `loss = BCE (con pos_weight)`
  - `lr = 1.5e-4`
  - `dropout = 0.2`
  - `flip_prob = 0.15`
  - `noise_std = 0.005`
  - `seed = 42`
- Objetivo de clasificación usado en este mejor resultado:
  - **triage**: positivo = `PC + EB`, negativo = resto.
- Resultado en test @0.5:
  - `accuracy = {tess_metrics['test_0_5']['accuracy']:.6f}`
  - `precision = {tess_metrics['test_0_5']['precision']:.6f}`
  - `recall = {tess_metrics['test_0_5']['recall']:.6f}`
  - `f1 = {tess_metrics['test_0_5']['f1']:.6f}`
  - `mcc = {tess_metrics['test_0_5']['mcc']:.6f}`
  - `ap = {tess_metrics['test_ap']:.6f}`

## Nota de comparabilidad
- Kepler es comparable de forma directa con el paper.
- TESS en esta carpeta guarda el mejor resultado **triage (PC+EB vs resto)**; el paper reporta métricas para **PC vs no-PC**. Por eso la comparación en `mejores_vs_paper.csv` incluye una nota de objetivo.
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main():
    # Kepler champion
    k_data = load_kepler_splits()
    k_xg_tr, k_xl_tr, k_y_tr, k_xg_va, k_xl_va, k_y_va, k_xg_te, k_xl_te, k_y_te = k_data
    k_xg_tr = preprocess_views(k_xg_tr, "zscore")
    k_xl_tr = preprocess_views(k_xl_tr, "zscore")
    k_xg_va = preprocess_views(k_xg_va, "zscore")
    k_xl_va = preprocess_views(k_xl_va, "zscore")
    k_xg_te = preprocess_views(k_xg_te, "zscore")
    k_xl_te = preprocess_views(k_xl_te, "zscore")

    k_model, k_metrics = train_model(
        k_xg_tr,
        k_xl_tr,
        k_y_tr,
        k_xg_va,
        k_xl_va,
        k_y_va,
        k_xg_te,
        k_xl_te,
        k_y_te,
        gl=2001,
        ll=201,
        drop=0.1,
        lr=2e-4,
        weight_decay=1e-5,
        epochs=34,
        patience=8,
        flip_prob=0.3,
        noise_std=0.005,
        use_pos_weight=False,
        seed=29,
    )
    torch.save(k_model.state_dict(), MODELS_DIR / "kepler_mejor_modelo.pth")
    k_payload = {
        "dataset": "Kepler",
        "objective": "PC vs no-PC",
        "config": {
            "preprocess": "zscore",
            "loss": "bce",
            "lr": 2e-4,
            "dropout": 0.1,
            "flip_prob": 0.3,
            "noise_std": 0.005,
            "use_pos_weight": False,
            "seed": 29,
        },
        **k_metrics,
    }
    with open(METRICS_DIR / "kepler_mejor_metrics.json", "w", encoding="utf-8") as f:
        json.dump(k_payload, f, indent=2, ensure_ascii=False)

    # TESS champion (triage objective)
    t_data = load_tess_triage_splits()
    t_xg_tr, t_xl_tr, t_y_tr, t_xg_va, t_xl_va, t_y_va, t_xg_te, t_xl_te, t_y_te = t_data
    t_xg_tr = preprocess_views(t_xg_tr, "robust")
    t_xl_tr = preprocess_views(t_xl_tr, "robust")
    t_xg_va = preprocess_views(t_xg_va, "robust")
    t_xl_va = preprocess_views(t_xl_va, "robust")
    t_xg_te = preprocess_views(t_xg_te, "robust")
    t_xl_te = preprocess_views(t_xl_te, "robust")

    t_model, t_metrics = train_model(
        t_xg_tr,
        t_xl_tr,
        t_y_tr,
        t_xg_va,
        t_xl_va,
        t_y_va,
        t_xg_te,
        t_xl_te,
        t_y_te,
        gl=201,
        ll=61,
        drop=0.2,
        lr=1.5e-4,
        weight_decay=1e-5,
        epochs=40,
        patience=10,
        flip_prob=0.15,
        noise_std=0.005,
        use_pos_weight=True,
        seed=42,
    )
    torch.save(t_model.state_dict(), MODELS_DIR / "tess_mejor_modelo.pth")
    t_payload = {
        "dataset": "TESS",
        "objective": "Triage: (PC + EB) vs resto",
        "config": {
            "preprocess": "robust",
            "loss": "bce",
            "lr": 1.5e-4,
            "dropout": 0.2,
            "flip_prob": 0.15,
            "noise_std": 0.005,
            "use_pos_weight": True,
            "seed": 42,
        },
        **t_metrics,
    }
    with open(METRICS_DIR / "tess_mejor_metrics.json", "w", encoding="utf-8") as f:
        json.dump(t_payload, f, indent=2, ensure_ascii=False)

    # CSV comparativo
    rows = [
        {
            "dataset": "Kepler",
            "source": "paper_xie_2025",
            "objective": "PC vs no-PC",
            "threshold": 0.5,
            "accuracy": 0.962,
            "precision": 0.890,
            "recall": 0.950,
            "f1": 0.957,
            "mcc": 0.894,
            "ap": np.nan,
            "notes": "Reported in paper table (threshold 0.5)",
        },
        {
            "dataset": "Kepler",
            "source": "mejor_local",
            "objective": "PC vs no-PC",
            "threshold": 0.5,
            "accuracy": k_metrics["test_0_5"]["accuracy"],
            "precision": k_metrics["test_0_5"]["precision"],
            "recall": k_metrics["test_0_5"]["recall"],
            "f1": k_metrics["test_0_5"]["f1"],
            "mcc": k_metrics["test_0_5"]["mcc"],
            "ap": k_metrics["test_ap"],
            "notes": "Champion local config (zscore + aug)",
        },
        {
            "dataset": "TESS",
            "source": "paper_xie_2025",
            "objective": "PC vs no-PC",
            "threshold": 0.5,
            "accuracy": 0.999,
            "precision": 0.970,
            "recall": 1.000,
            "f1": 0.995,
            "mcc": 0.979,
            "ap": np.nan,
            "notes": "Reported in paper table (threshold 0.5)",
        },
        {
            "dataset": "TESS",
            "source": "mejor_local",
            "objective": "(PC+EB) vs resto (triage)",
            "threshold": 0.5,
            "accuracy": t_metrics["test_0_5"]["accuracy"],
            "precision": t_metrics["test_0_5"]["precision"],
            "recall": t_metrics["test_0_5"]["recall"],
            "f1": t_metrics["test_0_5"]["f1"],
            "mcc": t_metrics["test_0_5"]["mcc"],
            "ap": t_metrics["test_ap"],
            "notes": "Objective differs from paper (triage objective)",
        },
    ]
    pd.DataFrame(rows).to_csv(OUT_DIR / "mejores_vs_paper.csv", index=False)

    # README
    build_readme(k_metrics, t_metrics)

    # resumen json
    summary = {
        "kepler": {
            "model": str((MODELS_DIR / "kepler_mejor_modelo.pth").resolve()),
            "metrics": str((METRICS_DIR / "kepler_mejor_metrics.json").resolve()),
        },
        "tess": {
            "model": str((MODELS_DIR / "tess_mejor_modelo.pth").resolve()),
            "metrics": str((METRICS_DIR / "tess_mejor_metrics.json").resolve()),
        },
        "comparison_csv": str((OUT_DIR / "mejores_vs_paper.csv").resolve()),
    }
    with open(OUT_DIR / "index.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Saved folder:", OUT_DIR)


if __name__ == "__main__":
    main()
