from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    global_len: int
    local_len: int
    dropout: float
    preprocess: str
    model_file: str
    metrics_file: str
    objective: str


DATASETS = {
    "kepler": DatasetSpec(
        name="kepler",
        global_len=2001,
        local_len=201,
        dropout=0.1,
        preprocess="zscore",
        model_file="kepler_mejor_modelo.pth",
        metrics_file="kepler_mejor_metrics.json",
        objective="PC vs no-PC",
    ),
    "tess": DatasetSpec(
        name="tess",
        global_len=201,
        local_len=61,
        dropout=0.2,
        preprocess="robust",
        model_file="tess_mejor_modelo.pth",
        metrics_file="tess_mejor_metrics.json",
        objective="Triage: PC+EB vs resto",
    ),
}

DELETION_FRACTIONS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def normalize_rows_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = np.nanmin(x, axis=1, keepdims=True)
    hi = np.nanmax(x, axis=1, keepdims=True)
    denom = hi - lo
    out = np.divide(x - lo, denom, out=np.zeros_like(x, dtype=np.float32), where=denom > 1e-12)
    return out.astype(np.float32)


def normalized_mass(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 0.0)
    total = float(x.sum())
    if not np.isfinite(total) or total <= 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return x / total


def transit_window(length: int, fraction: float = 0.16) -> tuple[int, int]:
    width = max(3, int(round(length * fraction)))
    center = length // 2
    start = max(0, center - width // 2)
    end = min(length, center + width // 2 + 1)
    return start, end


def importance_summary(heat: np.ndarray, window: tuple[int, int]) -> dict[str, float | int]:
    mass = normalized_mass(heat)
    start, end = window
    peak_idx = int(np.argmax(heat)) if len(heat) else -1
    center = (len(heat) - 1) / 2.0 if len(heat) > 1 else 0.0
    return {
        "peak_index": peak_idx,
        "peak_relative_position": float(peak_idx / max(len(heat) - 1, 1)),
        "peak_distance_to_center": float(abs(peak_idx - center)),
        "central_importance_ratio": float(mass[start:end].sum()),
    }


def topk_overlap(a: np.ndarray, b: np.ndarray, frac: float = 0.10) -> float:
    k = max(1, int(round(min(len(a), len(b)) * frac)))
    top_a = set(np.argsort(a)[-k:].tolist())
    top_b = set(np.argsort(b)[-k:].tolist())
    return float(len(top_a & top_b) / k)


def smooth_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(x, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(np.asarray(x, dtype=np.float32), kernel, mode="same").astype(np.float32)


def target_logit(logits: torch.Tensor, target_label: int) -> torch.Tensor:
    return logits[:, 0] if int(target_label) == 1 else -logits[:, 0]


def target_confidence_from_prob(prob: np.ndarray | float, target_label: int) -> np.ndarray | float:
    return prob if int(target_label) == 1 else 1.0 - prob


def consensus_importance(*maps: np.ndarray) -> np.ndarray:
    return normalize_01(np.mean([normalize_01(m) for m in maps], axis=0))


def preprocess_views(x: np.ndarray, mode: str) -> np.ndarray:
    x = x.astype(np.float32)
    if mode == "none":
        return x
    if mode == "zscore":
        mu = np.mean(x, axis=1, keepdims=True)
        sd = np.std(x, axis=1, keepdims=True) + 1e-6
        return ((x - mu) / sd).astype(np.float32)
    if mode == "robust":
        med = np.median(x, axis=1, keepdims=True)
        sd = np.std(x, axis=1, keepdims=True) + 1e-6
        z = (x - med) / sd
        return np.clip(z, -5.0, 5.0).astype(np.float32)
    raise ValueError(f"Preprocesamiento no soportado: {mode}")


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _ = x.size()
        y = self.squeeze(x).view(batch, channels)
        y = self.excitation(y).view(batch, channels, 1)
        return x * y.expand_as(x)


class ResNetUnit(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.d1 = nn.Linear(n, n)
        self.d2 = nn.Linear(n, n)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.d1(x))
        y = self.d2(y)
        return self.act(y + x)


class Net(nn.Module):
    def __init__(self, gl: int, ll: int, drop: float):
        super().__init__()
        self.gb = nn.Sequential(
            nn.Conv1d(1, 16, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool1d(5, 2),
            nn.Conv1d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool1d(5, 2),
            nn.Conv1d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool1d(5, 2),
            nn.Conv1d(64, 128, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool1d(5, 2),
            nn.Conv1d(128, 256, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(256, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool1d(5, 2),
        )
        self.lb = nn.Sequential(
            nn.Conv1d(1, 16, 5, 1, 2),
            nn.ReLU(),
            SEBlock1D(16),
            nn.Conv1d(16, 16, 5, 1, 2),
            nn.ReLU(),
            SEBlock1D(16),
            nn.MaxPool1d(5, 2),
            nn.Conv1d(16, 32, 5, 1, 2),
            nn.ReLU(),
            SEBlock1D(32),
            nn.Conv1d(32, 32, 5, 1, 2),
            nn.ReLU(),
            SEBlock1D(32),
            nn.MaxPool1d(5, 2),
        )
        with torch.no_grad():
            f = self.gb(torch.zeros(1, 1, gl)).numel() + self.lb(torch.zeros(1, 1, ll)).numel()
        self.fc = nn.Linear(f, 512)
        self.drop = nn.Dropout(drop)
        self.r = nn.Sequential(*[ResNetUnit(512) for _ in range(6)])
        self.o = nn.Linear(512, 1)

    def forward(self, xg: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        g = self.gb(xg).reshape(xg.size(0), -1)
        l = self.lb(xl).reshape(xl.size(0), -1)
        z = torch.cat((g, l), 1)
        z = F.leaky_relu(self.fc(z), 0.1)
        z = self.drop(z)
        z = self.r(z)
        return self.o(z)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_state_dict(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(root: Path, spec: DatasetSpec, device: torch.device) -> Net:
    model = Net(spec.global_len, spec.local_len, spec.dropout).to(device)
    model_path = root / "mejores_resultados" / "modelos" / spec.model_file
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el checkpoint: {model_path}")
    state = load_state_dict(model_path, device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_threshold(root: Path, spec: DatasetSpec) -> float:
    metrics_path = root / "mejores_resultados" / "metricas" / spec.metrics_file
    if not metrics_path.exists():
        return 0.5
    metrics = load_json(metrics_path)
    return float(metrics.get("best_thr", 0.5))


def load_kepler_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    h5_path = root / "datos_procesados_h5" / "kepler_dataset.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"No existe el dataset Kepler: {h5_path}")
    with h5py.File(h5_path, "r") as hf:
        if split not in hf:
            raise KeyError(f"Split Kepler no encontrado: {split}")
        xg = hf[split]["global_view"][:].astype(np.float32)
        xl = hf[split]["local_view"][:].astype(np.float32)
        y = hf[split]["labels"][:].astype(np.int32)
    raw = ["PC" if int(v) == 1 else "no-PC" for v in y]
    return xg, xl, y, raw


def load_kepler_test(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    return load_kepler_split(root, "test")


def load_tess_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow es necesario para leer los TFRecords de TESS. "
            "Instalalo con `pip install tensorflow` o usa el entorno del proyecto."
        ) from exc

    base = root / "tfrecords_TESS"
    files = sorted(glob.glob(str(base / f"{split}-*")))
    if not files:
        raise FileNotFoundError(f"No se encontraron TFRecords {split}-* en {base}")

    rows = []
    for rec in tf.data.TFRecordDataset(files):
        ex = tf.train.Example()
        ex.ParseFromString(rec.numpy())
        f = ex.features.feature
        disp = f["Disposition"].bytes_list.value[0].decode()
        g = np.array(f["global_view"].float_list.value, dtype=np.float32)
        l = np.array(f["local_view"].float_list.value, dtype=np.float32)
        h = hashlib.sha1(np.concatenate([g, l]).tobytes()).hexdigest()
        rows.append((h, disp, g, l))

    seen = set()
    dedup = []
    for h, disp, g, l in rows:
        if h in seen:
            continue
        seen.add(h)
        dedup.append((disp, g, l))

    xg = np.stack([r[1] for r in dedup], axis=0).astype(np.float32)
    xl = np.stack([r[2] for r in dedup], axis=0).astype(np.float32)
    y = np.array([1 if r[0] in {"PC", "EB"} else 0 for r in dedup], dtype=np.int32)
    raw = [r[0] for r in dedup]
    return xg, xl, y, raw


def load_tess_test(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    return load_tess_split(root, "test")


def load_dataset_split(root: Path, spec: DatasetSpec, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if spec.name == "kepler":
        xg, xl, y, raw = load_kepler_split(root, split)
    elif spec.name == "tess":
        xg, xl, y, raw = load_tess_split(root, split)
    else:
        raise ValueError(spec.name)
    return preprocess_views(xg, spec.preprocess), preprocess_views(xl, spec.preprocess), y, raw


def load_dataset(root: Path, spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    return load_dataset_split(root, spec, "test")


def iter_batches(xg: np.ndarray, xl: np.ndarray, batch_size: int):
    for start in range(0, len(xg), batch_size):
        end = min(start + batch_size, len(xg))
        bg = torch.from_numpy(xg[start:end]).unsqueeze(1)
        bl = torch.from_numpy(xl[start:end]).unsqueeze(1)
        yield start, end, bg, bl


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))).astype(np.float32)


def logit_np(p: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def calibrate_probabilities(probs: np.ndarray, temperature: float) -> np.ndarray:
    return sigmoid_np(logit_np(probs) / max(float(temperature), 1e-6))


def predict_logits(
    model: Net,
    xg: np.ndarray,
    xl: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    logits_out = []
    with torch.no_grad():
        for _, _, bg, bl in iter_batches(xg, xl, batch_size):
            logits = model(bg.to(device), bl.to(device))
            logits_out.append(logits.cpu().numpy().ravel())
    return np.concatenate(logits_out).astype(np.float32)


def predict_probs(
    model: Net,
    xg: np.ndarray,
    xl: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    return sigmoid_np(predict_logits(model, xg, xl, device, batch_size=batch_size))


def fit_temperature(logits: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    if len(np.unique(y)) < 2:
        return 1.0
    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
    y_t = torch.tensor(y.astype(np.float32), dtype=torch.float32, device=device)
    log_temperature = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=80, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temperature).clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(logits_t / temperature, y_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return float(torch.exp(log_temperature).clamp(0.05, 20.0).cpu())


def calibration_metrics(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> tuple[dict, list[dict]]:
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin": i,
                    "bin_start": float(lo),
                    "bin_end": float(hi),
                    "count": 0,
                    "confidence": np.nan,
                    "accuracy": np.nan,
                    "gap": np.nan,
                }
            )
            continue
        confidence = float(probs[mask].mean())
        accuracy = float(y[mask].mean())
        gap = abs(confidence - accuracy)
        ece += (count / len(probs)) * gap
        mce = max(mce, gap)
        rows.append(
            {
                "bin": i,
                "bin_start": float(lo),
                "bin_end": float(hi),
                "count": count,
                "confidence": confidence,
                "accuracy": accuracy,
                "gap": float(gap),
            }
        )
    metrics = {
        "n": int(len(probs)),
        "ece": float(ece),
        "mce": float(mce),
        "brier": float(np.mean((probs - y) ** 2)),
        "nll": float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))),
    }
    return metrics, rows


def activate_dropout_only(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_probs(
    model: Net,
    xg: np.ndarray,
    xl: np.ndarray,
    device: torch.device,
    samples: int,
    batch_size: int = 256,
) -> np.ndarray:
    draws = []
    with torch.no_grad():
        for _ in range(samples):
            activate_dropout_only(model)
            probs = []
            for _, _, bg, bl in iter_batches(xg, xl, batch_size):
                logits = model(bg.to(device), bl.to(device))
                probs.append(torch.sigmoid(logits).cpu().numpy().ravel())
            draws.append(np.concatenate(probs).astype(np.float32))
    model.eval()
    return np.stack(draws, axis=0)


def compute_saliency(
    model: Net,
    xg: np.ndarray,
    xl: np.ndarray,
    device: torch.device,
    target_label: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    xgt = torch.tensor(xg[None, None, :], dtype=torch.float32, device=device, requires_grad=True)
    xlt = torch.tensor(xl[None, None, :], dtype=torch.float32, device=device, requires_grad=True)
    model.zero_grad(set_to_none=True)
    logits = model(xgt, xlt)
    target_logit(logits, target_label).sum().backward()
    sg = xgt.grad.detach().abs().squeeze().cpu().numpy()
    sl = xlt.grad.detach().abs().squeeze().cpu().numpy()
    return normalize_01(sg), normalize_01(sl)


def compute_batched_saliency(
    model: Net,
    xg: np.ndarray,
    xl: np.ndarray,
    target_labels: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    sal_g = []
    sal_l = []
    for start, end, bg, bl in iter_batches(xg, xl, batch_size):
        xgt = bg.to(device).requires_grad_(True)
        xlt = bl.to(device).requires_grad_(True)
        labels = torch.tensor(target_labels[start:end], dtype=torch.float32, device=device)
        signs = torch.where(labels > 0.5, torch.ones_like(labels), -torch.ones_like(labels))
        model.zero_grad(set_to_none=True)
        logits = model(xgt, xlt)[:, 0]
        (logits * signs).sum().backward()
        sal_g.append(xgt.grad.detach().abs().squeeze(1).cpu().numpy())
        sal_l.append(xlt.grad.detach().abs().squeeze(1).cpu().numpy())
    return normalize_rows_01(np.concatenate(sal_g, axis=0)), normalize_rows_01(np.concatenate(sal_l, axis=0))


def connected_component_lengths(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        empty = np.array([], dtype=np.int32)
        return empty, empty, empty
    padded = np.concatenate([[False], mask, [False]])
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    lengths = ends - starts
    return starts.astype(np.int32), ends.astype(np.int32), lengths.astype(np.int32)


def attention_focus_stats(heat: np.ndarray) -> dict[str, float | int]:
    heat = normalize_01(np.asarray(heat, dtype=np.float32))
    n = len(heat)
    if n == 0 or float(np.max(heat)) <= 1e-12:
        return {
            "focus_count": 0,
            "central_focus_count": 0,
            "central_focus_fraction": 0.0,
            "mean_focus_width": 0.0,
            "max_focus_width": 0.0,
            "focus_density": 0.0,
            "attention_position_variance": 0.0,
            "attention_entropy": 0.0,
            "central_importance_ratio": 0.0,
            "peak_distance_to_center": 0.0,
        }

    smooth_window = max(1, int(round(n * 0.015)))
    if smooth_window % 2 == 0:
        smooth_window += 1
    smoothed = normalize_01(smooth_1d(heat, smooth_window))
    threshold = max(float(np.quantile(smoothed, 0.90)), 0.35)
    starts, ends, lengths = connected_component_lengths(smoothed >= threshold)

    min_width = max(1, int(round(n * 0.01)))
    keep = lengths >= min_width
    starts = starts[keep]
    ends = ends[keep]
    lengths = lengths[keep]

    central_start, central_end = transit_window(n)
    overlaps_center = (starts < central_end) & (ends > central_start) if len(starts) else np.array([], dtype=bool)
    mass = normalized_mass(smoothed)
    positions = np.linspace(0.0, 1.0, n, dtype=np.float64)
    mean_pos = float(np.sum(mass * positions))
    pos_var = float(np.sum(mass * (positions - mean_pos) ** 2))
    entropy = float(-(mass * np.log(mass + 1e-12)).sum() / np.log(max(n, 2)))
    peak_idx = int(np.argmax(smoothed))
    center = (n - 1) / 2.0

    return {
        "focus_count": int(len(starts)),
        "central_focus_count": int(overlaps_center.sum()) if len(starts) else 0,
        "central_focus_fraction": float(overlaps_center.mean()) if len(starts) else 0.0,
        "mean_focus_width": float(lengths.mean()) if len(lengths) else 0.0,
        "max_focus_width": float(lengths.max()) if len(lengths) else 0.0,
        "focus_density": float(lengths.sum() / n) if len(lengths) else 0.0,
        "attention_position_variance": pos_var,
        "attention_entropy": entropy,
        "central_importance_ratio": float(mass[central_start:central_end].sum()),
        "peak_distance_to_center": float(abs(peak_idx - center)),
    }


def build_attention_focus_rows(
    dataset: str,
    y: np.ndarray,
    pred_labels: np.ndarray,
    raw_labels: list[str],
    probs: np.ndarray,
    mc_std: np.ndarray,
    sal_g: np.ndarray,
    sal_l: np.ndarray,
) -> list[dict]:
    outcomes = outcome_labels(y, pred_labels)
    rows = []
    for idx in range(len(y)):
        g_stats = attention_focus_stats(sal_g[idx])
        l_stats = attention_focus_stats(sal_l[idx])
        rows.append(
            {
                "dataset": dataset,
                "index": int(idx),
                "raw_label": raw_labels[idx],
                "y_true": int(y[idx]),
                "pred_label": int(pred_labels[idx]),
                "outcome": str(outcomes[idx]),
                "probability": float(probs[idx]),
                "target_confidence": float(target_confidence_from_prob(float(probs[idx]), int(pred_labels[idx]))),
                "mc_std": float(mc_std[idx]),
                "global_focus_count": int(g_stats["focus_count"]),
                "local_focus_count": int(l_stats["focus_count"]),
                "total_focus_count": int(g_stats["focus_count"]) + int(l_stats["focus_count"]),
                "global_central_focus_count": int(g_stats["central_focus_count"]),
                "local_central_focus_count": int(l_stats["central_focus_count"]),
                "global_central_focus_fraction": float(g_stats["central_focus_fraction"]),
                "local_central_focus_fraction": float(l_stats["central_focus_fraction"]),
                "global_focus_density": float(g_stats["focus_density"]),
                "local_focus_density": float(l_stats["focus_density"]),
                "global_attention_position_variance": float(g_stats["attention_position_variance"]),
                "local_attention_position_variance": float(l_stats["attention_position_variance"]),
                "mean_attention_position_variance": float(
                    0.5 * (g_stats["attention_position_variance"] + l_stats["attention_position_variance"])
                ),
                "global_attention_entropy": float(g_stats["attention_entropy"]),
                "local_attention_entropy": float(l_stats["attention_entropy"]),
                "mean_attention_entropy": float(0.5 * (g_stats["attention_entropy"] + l_stats["attention_entropy"])),
                "global_central_importance_ratio": float(g_stats["central_importance_ratio"]),
                "local_central_importance_ratio": float(l_stats["central_importance_ratio"]),
                "mean_central_importance_ratio": float(
                    0.5 * (g_stats["central_importance_ratio"] + l_stats["central_importance_ratio"])
                ),
                "global_peak_distance_to_center": float(g_stats["peak_distance_to_center"]),
                "local_peak_distance_to_center": float(l_stats["peak_distance_to_center"]),
            }
        )
    return rows


def compute_smoothgrad(
    model: Net,
    xg: np.ndarray,
    xl: np.ndarray,
    device: torch.device,
    target_label: int,
    samples: int = 32,
    noise_std: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    xg_base = torch.tensor(xg[None, None, :], dtype=torch.float32, device=device)
    xl_base = torch.tensor(xl[None, None, :], dtype=torch.float32, device=device)
    xgt = (xg_base.repeat(samples, 1, 1) + noise_std * torch.randn(samples, 1, len(xg), device=device)).requires_grad_(True)
    xlt = (xl_base.repeat(samples, 1, 1) + noise_std * torch.randn(samples, 1, len(xl), device=device)).requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(xgt, xlt)
    target_logit(logits, target_label).sum().backward()
    sg = xgt.grad.detach().abs().mean(dim=0).squeeze().cpu().numpy()
    sl = xlt.grad.detach().abs().mean(dim=0).squeeze().cpu().numpy()
    return normalize_01(sg), normalize_01(sl)


def compute_integrated_gradients(
    model: Net,
    xg: np.ndarray,
    xl: np.ndarray,
    device: torch.device,
    target_label: int,
    steps: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    alphas = torch.linspace(0.0, 1.0, steps, device=device).view(steps, 1, 1)
    xg_input = torch.tensor(xg[None, None, :], dtype=torch.float32, device=device)
    xl_input = torch.tensor(xl[None, None, :], dtype=torch.float32, device=device)
    xg_scaled = (alphas * xg_input).detach().requires_grad_(True)
    xl_scaled = (alphas * xl_input).detach().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    logits = model(xg_scaled, xl_scaled)[:, 0]
    signed_logits = logits if int(target_label) == 1 else -logits
    signed_logits.sum().backward()

    grad_g = xg_scaled.grad.detach().mean(dim=0).squeeze(0)
    grad_l = xl_scaled.grad.detach().mean(dim=0).squeeze(0)
    attr_g = (xg_input.detach().squeeze(0).squeeze(0) * grad_g).abs().cpu().numpy()
    attr_l = (xl_input.detach().squeeze(0).squeeze(0) * grad_l).abs().cpu().numpy()
    return normalize_01(attr_g), normalize_01(attr_l)


def occlusion_window(length: int) -> tuple[int, int]:
    window = max(4, int(round(length * 0.06)))
    stride = max(1, window // 4)
    return window, stride


def compute_occlusion_sensitivity(
    model: Net,
    xg: np.ndarray,
    xl: np.ndarray,
    device: torch.device,
    target_label: int,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    xg_base = torch.tensor(xg[None, None, :], dtype=torch.float32, device=device)
    xl_base = torch.tensor(xl[None, None, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        base_prob = float(torch.sigmoid(model(xg_base, xl_base))[0, 0].cpu())
        base_conf = float(target_confidence_from_prob(base_prob, target_label))

    def run_view(view: str) -> np.ndarray:
        length = len(xg) if view == "global" else len(xl)
        window, stride = occlusion_window(length)
        starts = list(range(0, max(1, length - window + 1), stride))
        if starts[-1] != length - window:
            starts.append(max(0, length - window))

        heat = np.zeros(length, dtype=np.float64)
        counts = np.zeros(length, dtype=np.float64)
        deltas = []
        ranges = []
        for start in starts:
            end = min(length, start + window)
            ranges.append((start, end))
            gx = xg.copy()
            lx = xl.copy()
            if view == "global":
                gx[start:end] = 0.0
            else:
                lx[start:end] = 0.0
            deltas.append((gx, lx))

        confidences = []
        with torch.no_grad():
            for offset in range(0, len(deltas), batch_size):
                chunk = deltas[offset : offset + batch_size]
                bg = torch.tensor(np.stack([d[0] for d in chunk])[:, None, :], dtype=torch.float32, device=device)
                bl = torch.tensor(np.stack([d[1] for d in chunk])[:, None, :], dtype=torch.float32, device=device)
                probs = torch.sigmoid(model(bg, bl)).cpu().numpy().ravel()
                confidences.extend(np.asarray(target_confidence_from_prob(probs, target_label)).ravel().tolist())

        for (start, end), conf in zip(ranges, confidences):
            delta = max(0.0, base_conf - float(conf))
            heat[start:end] += delta
            counts[start:end] += 1.0
        heat = np.divide(heat, np.maximum(counts, 1.0))
        return normalize_01(heat)

    return run_view("global"), run_view("local")


def last_conv(module: nn.Module) -> nn.Conv1d:
    convs = [m for m in module.modules() if isinstance(m, nn.Conv1d)]
    if not convs:
        raise ValueError("No se encontro Conv1d para Grad-CAM.")
    return convs[-1]


def conv_layers(module: nn.Module) -> list[nn.Conv1d]:
    layers = [m for m in module.modules() if isinstance(m, nn.Conv1d)]
    if not layers:
        raise ValueError("No se encontraron capas Conv1d.")
    return layers


def compute_gradcam(
    model: Net,
    xg: np.ndarray,
    xl: np.ndarray,
    device: torch.device,
    target_label: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    layers = {
        "global": last_conv(model.gb),
        "local": last_conv(model.lb),
    }
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}
    handles = []

    def make_forward_hook(name: str):
        def hook(_module, _inp, out):
            activations[name] = out

        return hook

    def make_backward_hook(name: str):
        def hook(_module, _grad_in, grad_out):
            gradients[name] = grad_out[0]

        return hook

    for name, layer in layers.items():
        handles.append(layer.register_forward_hook(make_forward_hook(name)))
        handles.append(layer.register_full_backward_hook(make_backward_hook(name)))

    try:
        xgt = torch.tensor(xg[None, None, :], dtype=torch.float32, device=device, requires_grad=True)
        xlt = torch.tensor(xl[None, None, :], dtype=torch.float32, device=device, requires_grad=True)
        model.zero_grad(set_to_none=True)
        logits = model(xgt, xlt)
        target_logit(logits, target_label).sum().backward()

        cams = {}
        target_lengths = {"global": len(xg), "local": len(xl)}
        for name in ("global", "local"):
            acts = activations[name].detach()
            grads = gradients[name].detach()
            weights = grads.mean(dim=2, keepdim=True)
            cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=target_lengths[name], mode="linear", align_corners=False)
            cams[name] = normalize_01(cam.squeeze().cpu().numpy())
    finally:
        for handle in handles:
            handle.remove()

    return cams["global"], cams["local"]


def compute_multiscale_gradcam(
    model: Net,
    xg: np.ndarray,
    xl: np.ndarray,
    device: torch.device,
    target_label: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    branches = {
        "global": conv_layers(model.gb),
        "local": conv_layers(model.lb),
    }
    activations: dict[tuple[str, int], torch.Tensor] = {}
    gradients: dict[tuple[str, int], torch.Tensor] = {}
    handles = []

    def make_forward_hook(key: tuple[str, int]):
        def hook(_module, _inp, out):
            activations[key] = out

        return hook

    def make_backward_hook(key: tuple[str, int]):
        def hook(_module, _grad_in, grad_out):
            gradients[key] = grad_out[0]

        return hook

    for branch, layers in branches.items():
        for i, layer in enumerate(layers):
            key = (branch, i)
            handles.append(layer.register_forward_hook(make_forward_hook(key)))
            handles.append(layer.register_full_backward_hook(make_backward_hook(key)))

    try:
        xgt = torch.tensor(xg[None, None, :], dtype=torch.float32, device=device, requires_grad=True)
        xlt = torch.tensor(xl[None, None, :], dtype=torch.float32, device=device, requires_grad=True)
        model.zero_grad(set_to_none=True)
        logits = model(xgt, xlt)
        target_logit(logits, target_label).sum().backward()

        target_lengths = {"global": len(xg), "local": len(xl)}
        cams = {}
        for branch, layers in branches.items():
            maps = []
            for i, _layer in enumerate(layers):
                key = (branch, i)
                acts = activations[key].detach()
                grads = gradients[key].detach()
                # LayerCAM-style weighting keeps finer temporal evidence from early layers.
                cam = F.relu((F.relu(grads) * acts).sum(dim=1, keepdim=True))
                cam = F.interpolate(cam, size=target_lengths[branch], mode="linear", align_corners=False)
                maps.append(normalize_01(cam.squeeze().cpu().numpy()))
            cams[branch] = consensus_importance(*maps)
    finally:
        for handle in handles:
            handle.remove()

    return cams["global"], cams["local"]


def predict_one_prob(model: Net, xg: np.ndarray, xl: np.ndarray, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        bg = torch.tensor(xg[None, None, :], dtype=torch.float32, device=device)
        bl = torch.tensor(xl[None, None, :], dtype=torch.float32, device=device)
        return float(torch.sigmoid(model(bg, bl))[0, 0].cpu())


def delete_top_fraction(values: np.ndarray, heat: np.ndarray, fraction: float) -> np.ndarray:
    out = values.copy()
    if fraction <= 0.0:
        return out
    k = max(1, int(round(len(out) * fraction)))
    top_idx = np.argsort(heat)[-k:]
    out[top_idx] = 0.0
    return out


def build_deletion_rows(
    model: Net,
    dataset: str,
    case: str,
    idx: int,
    xg: np.ndarray,
    xl: np.ndarray,
    target_label: int,
    method_maps: dict[str, tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> list[dict]:
    base_prob = predict_one_prob(model, xg, xl, device)
    base_conf = float(target_confidence_from_prob(base_prob, target_label))
    rows = []
    for method, (heat_g, heat_l) in method_maps.items():
        for fraction in DELETION_FRACTIONS:
            xg_deleted = delete_top_fraction(xg, heat_g, fraction)
            xl_deleted = delete_top_fraction(xl, heat_l, fraction)
            prob = predict_one_prob(model, xg_deleted, xl_deleted, device)
            conf = float(target_confidence_from_prob(prob, target_label))
            rows.append(
                {
                    "dataset": dataset,
                    "case": case,
                    "index": idx,
                    "method": method,
                    "target_label": int(target_label),
                    "deleted_fraction": float(fraction),
                    "base_probability": float(base_prob),
                    "base_target_confidence": float(base_conf),
                    "probability_after_deletion": float(prob),
                    "target_confidence_after_deletion": float(conf),
                    "confidence_drop": float(base_conf - conf),
                }
            )
    return rows


def choose_examples(
    y: np.ndarray,
    probs: np.ndarray,
    mc_std: np.ndarray,
    threshold: float,
    max_examples: int,
) -> list[dict]:
    pred = (probs >= threshold).astype(np.int32)
    selected: list[dict] = []
    used: set[int] = set()

    def add(case: str, candidates: np.ndarray, order_values: np.ndarray, reverse: bool) -> None:
        if len(selected) >= max_examples or candidates.size == 0:
            return
        order = np.argsort(order_values)
        if reverse:
            order = order[::-1]
        for pos in order:
            idx = int(candidates[pos])
            if idx not in used:
                used.add(idx)
                selected.append({"case": case, "index": idx})
                return

    tp = np.where((y == 1) & (pred == 1))[0]
    tn = np.where((y == 0) & (pred == 0))[0]
    fp = np.where((y == 0) & (pred == 1))[0]
    fn = np.where((y == 1) & (pred == 0))[0]
    all_idx = np.arange(len(y))

    add("positivo_correcto_alta_confianza", tp, probs[tp], reverse=True)
    add("negativo_correcto_alta_confianza", tn, probs[tn], reverse=False)
    add("cerca_del_umbral", all_idx, np.abs(probs - threshold), reverse=False)
    add("falso_positivo", fp, probs[fp], reverse=True)
    add("falso_negativo", fn, probs[fn], reverse=False)
    add("mayor_incertidumbre_mc_dropout", all_idx, mc_std, reverse=True)
    return selected[:max_examples]


def plot_heatmap_line(
    ax: plt.Axes,
    curve: np.ndarray,
    heat: np.ndarray,
    title: str,
    ylabel: str,
) -> None:
    x = np.arange(len(curve))
    start, end = transit_window(len(curve))
    ymin = float(np.nanmin(curve))
    ymax = float(np.nanmax(curve))
    if abs(ymax - ymin) < 1e-6:
        ymin -= 1.0
        ymax += 1.0
    pad = 0.06 * (ymax - ymin)
    ax.imshow(
        heat[None, :],
        aspect="auto",
        cmap="magma",
        extent=[0, len(curve) - 1, ymin - pad, ymax + pad],
        origin="lower",
        alpha=0.42,
    )
    ax.axvspan(start, end - 1, color="#2f9e44", alpha=0.12, lw=0)
    ax.axvline(len(curve) // 2, color="#2f9e44", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.plot(x, curve, color="#101010", lw=0.85)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, len(curve) - 1)
    ax.grid(alpha=0.18, linewidth=0.6)


def plot_explanation(
    dataset: str,
    case: str,
    idx: int,
    xg: np.ndarray,
    xl: np.ndarray,
    sal_g: np.ndarray,
    sal_l: np.ndarray,
    smooth_g: np.ndarray,
    smooth_l: np.ndarray,
    ig_g: np.ndarray,
    ig_l: np.ndarray,
    cam_g: np.ndarray,
    cam_l: np.ndarray,
    mcam_g: np.ndarray,
    mcam_l: np.ndarray,
    occ_g: np.ndarray,
    occ_l: np.ndarray,
    cons_g: np.ndarray,
    cons_l: np.ndarray,
    y_true: int,
    prob: float,
    threshold: float,
    mc_mean: float,
    mc_std: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(7, 2, figsize=(15, 19), constrained_layout=True)
    fig.suptitle(
        (
            f"{dataset.upper()} | {case} | idx={idx} | y={y_true} | "
            f"p={prob:.3f} | thr={threshold:.2f} | MC={mc_mean:.3f}+/-{mc_std:.3f}"
        ),
        fontsize=12,
        fontweight="bold",
    )
    plot_heatmap_line(axes[0, 0], xg, sal_g, "Vista global - Saliency Map", "flujo")
    plot_heatmap_line(axes[0, 1], xl, sal_l, "Vista local - Saliency Map", "flujo")
    plot_heatmap_line(axes[1, 0], xg, smooth_g, "Vista global - SmoothGrad", "flujo")
    plot_heatmap_line(axes[1, 1], xl, smooth_l, "Vista local - SmoothGrad", "flujo")
    plot_heatmap_line(axes[2, 0], xg, ig_g, "Vista global - Integrated Gradients", "flujo")
    plot_heatmap_line(axes[2, 1], xl, ig_l, "Vista local - Integrated Gradients", "flujo")
    plot_heatmap_line(axes[3, 0], xg, occ_g, "Vista global - Occlusion Sensitivity", "flujo")
    plot_heatmap_line(axes[3, 1], xl, occ_l, "Vista local - Occlusion Sensitivity", "flujo")
    plot_heatmap_line(axes[4, 0], xg, cam_g, "Vista global - Grad-CAM 1D (diagnostico)", "flujo")
    plot_heatmap_line(axes[4, 1], xl, cam_l, "Vista local - Grad-CAM 1D (diagnostico)", "flujo")
    plot_heatmap_line(axes[5, 0], xg, mcam_g, "Vista global - Grad-CAM multiescala", "flujo")
    plot_heatmap_line(axes[5, 1], xl, mcam_l, "Vista local - Grad-CAM multiescala", "flujo")
    plot_heatmap_line(axes[6, 0], xg, cons_g, "Vista global - Consenso entrada", "flujo")
    plot_heatmap_line(axes[6, 1], xl, cons_l, "Vista local - Consenso entrada", "flujo")
    axes[6, 0].set_xlabel("indice temporal")
    axes[6, 1].set_xlabel("indice temporal")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_mc_uncertainty(
    dataset: str,
    example_rows: list[dict],
    threshold: float,
    out_path: Path,
) -> None:
    if not example_rows:
        return
    labels = [f"{r['case']} ({r['index']})" for r in example_rows]
    means = np.array([r["mc_mean"] for r in example_rows], dtype=np.float32)
    stds = np.array([r["mc_std"] for r in example_rows], dtype=np.float32)
    p05 = np.array([r["mc_p05"] for r in example_rows], dtype=np.float32)
    p95 = np.array([r["mc_p95"] for r in example_rows], dtype=np.float32)
    xerr = np.vstack([means - p05, p95 - means])

    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.55 * len(labels))), constrained_layout=True)
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, means, xerr=xerr, color="#3d7ea6", alpha=0.82, capsize=3)
    ax.scatter(means, y_pos, s=18 + 700 * stds, color="#d1495b", alpha=0.7, label="std MC")
    ax.axvline(threshold, color="#202020", linestyle="--", linewidth=1.1, label=f"umbral={threshold:.2f}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("probabilidad por MC Dropout")
    ax.set_title(f"{dataset.upper()} - Incertidumbre MC Dropout", fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_mc_scatter(dataset: str, mc_df: pd.DataFrame, threshold: float, out_path: Path) -> None:
    df = mc_df.copy()
    df["outcome"] = np.select(
        [
            (df["y_true"] == 1) & (df["pred_label"] == 1),
            (df["y_true"] == 0) & (df["pred_label"] == 0),
            (df["y_true"] == 0) & (df["pred_label"] == 1),
            (df["y_true"] == 1) & (df["pred_label"] == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="NA",
    )
    colors = {"TP": "#2a9d8f", "TN": "#457b9d", "FP": "#e76f51", "FN": "#c1121f", "NA": "#6c757d"}
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for outcome, group in df.groupby("outcome"):
        ax.scatter(
            group["probability"],
            group["mc_std"],
            s=18,
            alpha=0.55,
            color=colors.get(outcome, "#6c757d"),
            label=f"{outcome} (n={len(group)})",
            edgecolors="none",
        )
    ax.axvline(threshold, color="#202020", linestyle="--", linewidth=1.1, label=f"umbral={threshold:.2f}")
    ax.set_xlabel("probabilidad deterministica")
    ax.set_ylabel("incertidumbre MC Dropout (std)")
    ax.set_title(f"{dataset.upper()} - Probabilidad vs incertidumbre en test", fontweight="bold")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper center", ncols=3, fontsize=8)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_aggregate_importance(dataset: str, maps: list[dict], out_path: Path) -> None:
    if not maps:
        return

    def mean_map(key: str) -> np.ndarray:
        arr = np.stack([m[key] for m in maps], axis=0)
        return normalize_01(arr.mean(axis=0))

    panels = [
        ("sal_g", "Global - Saliency promedio"),
        ("sal_l", "Local - Saliency promedio"),
        ("smooth_g", "Global - SmoothGrad promedio"),
        ("smooth_l", "Local - SmoothGrad promedio"),
        ("ig_g", "Global - Integrated Gradients promedio"),
        ("ig_l", "Local - Integrated Gradients promedio"),
        ("occ_g", "Global - Occlusion promedio"),
        ("occ_l", "Local - Occlusion promedio"),
        ("cam_g", "Global - Grad-CAM promedio"),
        ("cam_l", "Local - Grad-CAM promedio"),
        ("mcam_g", "Global - Grad-CAM multiescala promedio"),
        ("mcam_l", "Local - Grad-CAM multiescala promedio"),
        ("cons_g", "Global - Consenso promedio"),
        ("cons_l", "Local - Consenso promedio"),
    ]
    fig, axes = plt.subplots(7, 2, figsize=(15, 19), constrained_layout=True)
    fig.suptitle(f"{dataset.upper()} - Importancia promedio en ejemplos seleccionados", fontweight="bold")
    for ax, (key, title) in zip(axes.ravel(), panels):
        heat = mean_map(key)
        x = np.arange(len(heat))
        start, end = transit_window(len(heat))
        ax.fill_between(x, 0.0, heat, color="#3d7ea6", alpha=0.7)
        ax.plot(x, heat, color="#17324d", linewidth=1.2)
        ax.axvspan(start, end - 1, color="#2f9e44", alpha=0.14, lw=0)
        ax.axvline(len(heat) // 2, color="#2f9e44", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, len(heat) - 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("indice temporal")
        ax.set_ylabel("importancia normalizada")
        ax.grid(alpha=0.2)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_deletion_curves(dataset: str, rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    colors = {
        "saliency": "#457b9d",
        "smoothgrad": "#2a9d8f",
        "integrated_gradients": "#f4a261",
        "occlusion": "#e76f51",
        "gradcam": "#6d597a",
        "gradcam_multiscale": "#8d99ae",
        "consensus": "#1d3557",
    }
    for method, group in df.groupby("method"):
        curve = (
            group.groupby("deleted_fraction", as_index=False)["confidence_drop"]
            .mean()
            .sort_values("deleted_fraction")
        )
        ax.plot(
            100.0 * curve["deleted_fraction"],
            curve["confidence_drop"],
            marker="o",
            linewidth=1.8 if method == "consensus" else 1.25,
            color=colors.get(method, None),
            label=method,
        )
    ax.axhline(0.0, color="#202020", linewidth=0.9, alpha=0.6)
    ax.set_xlabel("porcentaje de puntos mas importantes borrados")
    ax.set_ylabel("caida media de confianza hacia la clase explicada")
    ax.set_title(f"{dataset.upper()} - Fidelidad por borrado", fontweight="bold")
    ax.grid(alpha=0.22)
    ax.legend(ncols=2, fontsize=8)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_reliability_diagram(dataset: str, rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color="#202020", linestyle="--", linewidth=1.0, label="calibracion ideal")
    for mode, group in df.groupby("calibration"):
        group = group.dropna(subset=["confidence", "accuracy"])
        ax.plot(
            group["confidence"],
            group["accuracy"],
            marker="o",
            linewidth=1.4,
            label=mode,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confianza promedio")
    ax.set_ylabel("fraccion positiva observada")
    ax.set_title(f"{dataset.upper()} - Diagrama de confiabilidad", fontweight="bold")
    ax.grid(alpha=0.22)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


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


def plot_outcome_uncertainty(dataset: str, mc_df: pd.DataFrame, out_path: Path) -> None:
    if mc_df.empty:
        return
    df = mc_df.copy()
    df["outcome"] = outcome_labels(df["y_true"].to_numpy(), df["pred_label"].to_numpy())
    summary = df.groupby("outcome", as_index=False).agg(
        n=("index", "count"),
        mean_mc_std=("mc_std", "mean"),
        mean_target_confidence=("target_confidence", "mean"),
    )
    order = ["TP", "TN", "FP", "FN"]
    summary["order"] = summary["outcome"].map({k: i for i, k in enumerate(order)}).fillna(99)
    summary = summary.sort_values("order")

    fig, ax1 = plt.subplots(figsize=(8, 5), constrained_layout=True)
    x = np.arange(len(summary))
    ax1.bar(x, summary["mean_mc_std"], color="#457b9d", alpha=0.82, label="std MC media")
    ax1.set_ylabel("incertidumbre MC media")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{r.outcome}\n(n={int(r.n)})" for r in summary.itertuples()], fontsize=9)
    ax1.grid(axis="y", alpha=0.22)
    ax2 = ax1.twinx()
    ax2.plot(x, summary["mean_target_confidence"], color="#e76f51", marker="o", label="confianza objetivo")
    ax2.set_ylabel("confianza objetivo media")
    ax2.set_ylim(0, 1)
    ax1.set_title(f"{dataset.upper()} - Incertidumbre por tipo de resultado", fontweight="bold")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def attention_confusion_summary(focus_df: pd.DataFrame) -> pd.DataFrame:
    if focus_df.empty:
        return pd.DataFrame()
    grouped = (
        focus_df.groupby(["dataset", "y_true", "pred_label", "outcome"], as_index=False)
        .agg(
            n=("index", "count"),
            total_focus_count_mean=("total_focus_count", "mean"),
            total_focus_count_std=("total_focus_count", "std"),
            total_focus_count_var=("total_focus_count", "var"),
            global_focus_count_mean=("global_focus_count", "mean"),
            local_focus_count_mean=("local_focus_count", "mean"),
            attention_position_variance_mean=("mean_attention_position_variance", "mean"),
            attention_position_variance_std=("mean_attention_position_variance", "std"),
            attention_entropy_mean=("mean_attention_entropy", "mean"),
            central_importance_ratio_mean=("mean_central_importance_ratio", "mean"),
            mc_std_mean=("mc_std", "mean"),
        )
        .fillna(0.0)
    )
    row_totals = grouped.groupby(["dataset", "y_true"])["n"].transform("sum").replace(0, np.nan)
    dataset_totals = grouped.groupby("dataset")["n"].transform("sum").replace(0, np.nan)
    grouped["row_total"] = row_totals.astype(float)
    grouped["row_normalized_rate"] = (grouped["n"] / row_totals).fillna(0.0)
    grouped["global_rate"] = (grouped["n"] / dataset_totals).fillna(0.0)
    return grouped.sort_values(["dataset", "y_true", "pred_label"]).reset_index(drop=True)


def attention_outcome_summary(focus_df: pd.DataFrame) -> pd.DataFrame:
    if focus_df.empty:
        return pd.DataFrame()
    return (
        focus_df.groupby(["dataset", "outcome"], as_index=False)
        .agg(
            n=("index", "count"),
            total_focus_count_mean=("total_focus_count", "mean"),
            total_focus_count_std=("total_focus_count", "std"),
            total_focus_count_var=("total_focus_count", "var"),
            global_focus_count_mean=("global_focus_count", "mean"),
            local_focus_count_mean=("local_focus_count", "mean"),
            attention_position_variance_mean=("mean_attention_position_variance", "mean"),
            attention_position_variance_std=("mean_attention_position_variance", "std"),
            attention_entropy_mean=("mean_attention_entropy", "mean"),
            central_importance_ratio_mean=("mean_central_importance_ratio", "mean"),
            mc_std_mean=("mc_std", "mean"),
            target_confidence_mean=("target_confidence", "mean"),
        )
        .fillna(0.0)
        .sort_values(["dataset", "outcome"])
    )


def plot_attention_focus_confusion_matrix(dataset: str, summary_df: pd.DataFrame, out_path: Path) -> None:
    if summary_df.empty:
        return
    df = summary_df[summary_df["dataset"] == dataset]
    if df.empty:
        return
    count_matrix = np.zeros((2, 2), dtype=np.float64)
    rate_matrix = np.zeros((2, 2), dtype=np.float64)
    labels = [["" for _ in range(2)] for _ in range(2)]
    if "row_normalized_rate" not in df.columns:
        df = df.copy()
        row_totals = df.groupby("y_true")["n"].transform("sum").replace(0, np.nan)
        df["row_normalized_rate"] = (df["n"] / row_totals).fillna(0.0)
    for row in df.itertuples(index=False):
        y_idx = int(row.y_true)
        p_idx = int(row.pred_label)
        rate = float(row.row_normalized_rate)
        count_matrix[y_idx, p_idx] = float(row.n)
        rate_matrix[y_idx, p_idx] = rate
        labels[y_idx][p_idx] = (
            f"{row.outcome}\n"
            f"{100.0 * rate:.1f}% de real {y_idx}\n"
            f"n={int(row.n)}\n"
            f"focos={row.total_focus_count_mean:.2f}+/-{row.total_focus_count_std:.2f}\n"
            f"disp={row.attention_position_variance_mean:.3f}"
        )

    fig, ax = plt.subplots(figsize=(8.8, 7.0), constrained_layout=True)
    im = ax.imshow(rate_matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred 0", "pred 1"])
    ax.set_yticklabels(["real 0", "real 1"])
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Etiqueta real")
    ax.set_title(f"{dataset.upper()} - Matriz de confusion normalizada + focos XAI", fontweight="bold")
    for i in range(2):
        for j in range(2):
            text = labels[i][j] or f"0.0% de real {i}\nn=0"
            color = "white" if rate_matrix[i, j] >= 0.55 else "#202020"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)
    fig.colorbar(im, ax=ax, label="proporcion normalizada por etiqueta real")
    ax.text(
        0.5,
        -0.16,
        "El color muestra porcentaje por fila real; el texto mantiene n y estadisticos de focos de atencion.",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#404040",
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_attention_focus_outcome_boxplot(dataset: str, focus_df: pd.DataFrame, out_path: Path) -> None:
    df = focus_df[focus_df["dataset"] == dataset].copy()
    if df.empty:
        return
    order = [x for x in ["TP", "TN", "FP", "FN"] if x in set(df["outcome"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    data_counts = [df.loc[df["outcome"] == outcome, "total_focus_count"].to_numpy() for outcome in order]
    data_var = [df.loc[df["outcome"] == outcome, "mean_attention_position_variance"].to_numpy() for outcome in order]
    axes[0].boxplot(data_counts, tick_labels=order, showmeans=True)
    axes[0].set_title("Cantidad de focos")
    axes[0].set_ylabel("focos global+local")
    axes[0].grid(axis="y", alpha=0.22)
    axes[1].boxplot(data_var, tick_labels=order, showmeans=True)
    axes[1].set_title("Dispersion temporal de atencion")
    axes[1].set_ylabel("varianza ponderada de posicion")
    axes[1].grid(axis="y", alpha=0.22)
    fig.suptitle(f"{dataset.upper()} - Distribucion de focos por resultado", fontweight="bold")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def counterfactual_variants(xg: np.ndarray, xl: np.ndarray) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    variants = []
    gw = transit_window(len(xg))
    lw = transit_window(len(xl))

    def flatten_center(arr: np.ndarray, window: tuple[int, int]) -> np.ndarray:
        out = arr.copy()
        start, end = window
        out[start:end] = 0.0
        return out

    def scale_dip(arr: np.ndarray, window: tuple[int, int], factor: float) -> np.ndarray:
        out = arr.copy()
        start, end = window
        segment = out[start:end]
        mask = segment < 0.0
        segment = segment.copy()
        segment[mask] = np.clip(segment[mask] * factor, -5.0, 5.0)
        out[start:end] = segment
        return out

    variants.append(("erase_global_transit", "Aplana ventana central global", flatten_center(xg, gw), xl.copy()))
    variants.append(("erase_local_transit", "Aplana ventana central local", xg.copy(), flatten_center(xl, lw)))
    variants.append(
        (
            "erase_both_transits",
            "Aplana ventanas centrales global y local",
            flatten_center(xg, gw),
            flatten_center(xl, lw),
        )
    )
    variants.append(("deepen_local_transit", "Profundiza valores negativos locales", xg.copy(), scale_dip(xl, lw, 1.5)))
    variants.append(("shallow_local_transit", "Reduce valores negativos locales", xg.copy(), scale_dip(xl, lw, 0.5)))
    variants.append(
        (
            "deepen_both_transits",
            "Profundiza valores negativos globales y locales",
            scale_dip(xg, gw, 1.5),
            scale_dip(xl, lw, 1.5),
        )
    )
    variants.append(
        (
            "shallow_both_transits",
            "Reduce valores negativos globales y locales",
            scale_dip(xg, gw, 0.5),
            scale_dip(xl, lw, 0.5),
        )
    )
    return variants


def build_counterfactual_rows(
    model: Net,
    dataset: str,
    case: str,
    idx: int,
    xg: np.ndarray,
    xl: np.ndarray,
    y_true: int,
    target_label: int,
    threshold: float,
    device: torch.device,
) -> list[dict]:
    base_prob = predict_one_prob(model, xg, xl, device)
    base_conf = float(target_confidence_from_prob(base_prob, target_label))
    rows = []
    for name, description, cf_g, cf_l in counterfactual_variants(xg, xl):
        cf_prob = predict_one_prob(model, cf_g, cf_l, device)
        cf_conf = float(target_confidence_from_prob(cf_prob, target_label))
        rows.append(
            {
                "dataset": dataset,
                "case": case,
                "index": idx,
                "counterfactual": name,
                "description": description,
                "y_true": int(y_true),
                "target_label": int(target_label),
                "threshold": float(threshold),
                "base_probability": float(base_prob),
                "counterfactual_probability": float(cf_prob),
                "probability_delta": float(cf_prob - base_prob),
                "base_target_confidence": float(base_conf),
                "counterfactual_target_confidence": float(cf_conf),
                "target_confidence_delta": float(cf_conf - base_conf),
                "pred_label_after": int(cf_prob >= threshold),
                "prediction_changed": int((cf_prob >= threshold) != bool(target_label)),
            }
        )
    return rows


def plot_counterfactual_effects(dataset: str, rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("counterfactual", as_index=False)["target_confidence_delta"]
        .mean()
        .sort_values("target_confidence_delta")
    )
    colors = ["#c1121f" if v < 0 else "#2a9d8f" for v in summary["target_confidence_delta"]]
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.barh(summary["counterfactual"], summary["target_confidence_delta"], color=colors, alpha=0.85)
    ax.axvline(0.0, color="#202020", linewidth=0.9)
    ax.set_xlabel("cambio medio en confianza hacia la clase explicada")
    ax.set_title(f"{dataset.upper()} - Efecto de contrafactuales simples", fontweight="bold")
    ax.grid(axis="x", alpha=0.22)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_importance_metric_rows(
    dataset: str,
    case: str,
    idx: int,
    sal_g: np.ndarray,
    sal_l: np.ndarray,
    smooth_g: np.ndarray,
    smooth_l: np.ndarray,
    ig_g: np.ndarray,
    ig_l: np.ndarray,
    cam_g: np.ndarray,
    cam_l: np.ndarray,
    mcam_g: np.ndarray,
    mcam_l: np.ndarray,
    occ_g: np.ndarray,
    occ_l: np.ndarray,
    cons_g: np.ndarray,
    cons_l: np.ndarray,
) -> list[dict]:
    rows = []
    views = {
        "global": {
            "saliency": sal_g,
            "smoothgrad": smooth_g,
            "integrated_gradients": ig_g,
            "gradcam": cam_g,
            "gradcam_multiscale": mcam_g,
            "occlusion": occ_g,
            "consensus": cons_g,
        },
        "local": {
            "saliency": sal_l,
            "smoothgrad": smooth_l,
            "integrated_gradients": ig_l,
            "gradcam": cam_l,
            "gradcam_multiscale": mcam_l,
            "occlusion": occ_l,
            "consensus": cons_l,
        },
    }
    for view, maps in views.items():
        sal = maps["saliency"]
        smooth = maps["smoothgrad"]
        ig = maps["integrated_gradients"]
        cam = maps["gradcam"]
        mcam = maps["gradcam_multiscale"]
        occ = maps["occlusion"]
        cons = maps["consensus"]
        sal_stats = importance_summary(sal, transit_window(len(sal)))
        smooth_stats = importance_summary(smooth, transit_window(len(smooth)))
        ig_stats = importance_summary(ig, transit_window(len(ig)))
        cam_stats = importance_summary(cam, transit_window(len(cam)))
        mcam_stats = importance_summary(mcam, transit_window(len(mcam)))
        occ_stats = importance_summary(occ, transit_window(len(occ)))
        cons_stats = importance_summary(cons, transit_window(len(cons)))
        rows.append(
            {
                "dataset": dataset,
                "case": case,
                "index": idx,
                "view": view,
                "saliency_peak_index": sal_stats["peak_index"],
                "saliency_peak_relative_position": sal_stats["peak_relative_position"],
                "saliency_peak_distance_to_center": sal_stats["peak_distance_to_center"],
                "saliency_central_importance_ratio": sal_stats["central_importance_ratio"],
                "smoothgrad_peak_index": smooth_stats["peak_index"],
                "smoothgrad_peak_relative_position": smooth_stats["peak_relative_position"],
                "smoothgrad_peak_distance_to_center": smooth_stats["peak_distance_to_center"],
                "smoothgrad_central_importance_ratio": smooth_stats["central_importance_ratio"],
                "integrated_gradients_peak_index": ig_stats["peak_index"],
                "integrated_gradients_peak_relative_position": ig_stats["peak_relative_position"],
                "integrated_gradients_peak_distance_to_center": ig_stats["peak_distance_to_center"],
                "integrated_gradients_central_importance_ratio": ig_stats["central_importance_ratio"],
                "gradcam_peak_index": cam_stats["peak_index"],
                "gradcam_peak_relative_position": cam_stats["peak_relative_position"],
                "gradcam_peak_distance_to_center": cam_stats["peak_distance_to_center"],
                "gradcam_central_importance_ratio": cam_stats["central_importance_ratio"],
                "gradcam_multiscale_peak_index": mcam_stats["peak_index"],
                "gradcam_multiscale_peak_relative_position": mcam_stats["peak_relative_position"],
                "gradcam_multiscale_peak_distance_to_center": mcam_stats["peak_distance_to_center"],
                "gradcam_multiscale_central_importance_ratio": mcam_stats["central_importance_ratio"],
                "occlusion_peak_index": occ_stats["peak_index"],
                "occlusion_peak_relative_position": occ_stats["peak_relative_position"],
                "occlusion_peak_distance_to_center": occ_stats["peak_distance_to_center"],
                "occlusion_central_importance_ratio": occ_stats["central_importance_ratio"],
                "consensus_peak_index": cons_stats["peak_index"],
                "consensus_peak_relative_position": cons_stats["peak_relative_position"],
                "consensus_peak_distance_to_center": cons_stats["peak_distance_to_center"],
                "consensus_central_importance_ratio": cons_stats["central_importance_ratio"],
                "top10_overlap_saliency_gradcam": topk_overlap(sal, cam, frac=0.10),
                "top10_overlap_gradcam_multiscale_occlusion": topk_overlap(mcam, occ, frac=0.10),
                "top10_overlap_gradcam_multiscale_integrated_gradients": topk_overlap(mcam, ig, frac=0.10),
                "top10_overlap_saliency_smoothgrad": topk_overlap(sal, smooth, frac=0.10),
                "top10_overlap_saliency_integrated_gradients": topk_overlap(sal, ig, frac=0.10),
                "top10_overlap_smoothgrad_integrated_gradients": topk_overlap(smooth, ig, frac=0.10),
                "top10_overlap_integrated_gradients_occlusion": topk_overlap(ig, occ, frac=0.10),
                "top10_overlap_consensus_integrated_gradients": topk_overlap(cons, ig, frac=0.10),
                "top10_overlap_consensus_occlusion": topk_overlap(cons, occ, frac=0.10),
            }
        )
    return rows


def run_dataset(
    root: Path,
    spec: DatasetSpec,
    device: torch.device,
    mc_samples: int,
    max_examples: int,
    batch_size: int,
    smooth_samples: int,
    ig_steps: int,
) -> tuple[list[dict], pd.DataFrame, list[dict], list[dict], list[dict], list[dict], list[dict], list[dict]]:
    figs_dir = root / "resultados_xai" / "figuras"
    figs_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(root, spec, device)
    xg, xl, y, raw_labels = load_dataset(root, spec)
    xg_val, xl_val, y_val, _raw_val = load_dataset_split(root, spec, "val")
    threshold = load_threshold(root, spec)
    val_logits = predict_logits(model, xg_val, xl_val, device=device, batch_size=batch_size)
    temperature = fit_temperature(val_logits, y_val, device=device)
    calibrated_threshold = float(sigmoid_np(np.array([float(logit_np(threshold)) / temperature]))[0])

    test_logits = predict_logits(model, xg, xl, device=device, batch_size=batch_size)
    probs = sigmoid_np(test_logits)
    calibrated_probs = sigmoid_np(test_logits / max(temperature, 1e-6))
    mc_draws = mc_dropout_probs(model, xg, xl, device=device, samples=mc_samples, batch_size=batch_size)
    mc_draws_calibrated = calibrate_probabilities(mc_draws, temperature)
    mc_mean = mc_draws.mean(axis=0)
    mc_std = mc_draws.std(axis=0)
    mc_p05 = np.percentile(mc_draws, 5, axis=0)
    mc_p95 = np.percentile(mc_draws, 95, axis=0)
    mc_mean_calibrated = mc_draws_calibrated.mean(axis=0)
    mc_std_calibrated = mc_draws_calibrated.std(axis=0)
    mc_p05_calibrated = np.percentile(mc_draws_calibrated, 5, axis=0)
    mc_p95_calibrated = np.percentile(mc_draws_calibrated, 95, axis=0)
    pred_labels = (probs >= threshold).astype(np.int32)
    target_confidences = np.where(pred_labels == 1, probs, 1.0 - probs)
    calibrated_target_confidences = np.where(pred_labels == 1, calibrated_probs, 1.0 - calibrated_probs)
    focus_sal_g, focus_sal_l = compute_batched_saliency(
        model,
        xg,
        xl,
        target_labels=pred_labels,
        device=device,
        batch_size=batch_size,
    )
    attention_focus_rows = build_attention_focus_rows(
        dataset=spec.name,
        y=y,
        pred_labels=pred_labels,
        raw_labels=raw_labels,
        probs=probs,
        mc_std=mc_std,
        sal_g=focus_sal_g,
        sal_l=focus_sal_l,
    )
    attention_focus_df = pd.DataFrame(attention_focus_rows)
    attention_confusion_df = attention_confusion_summary(attention_focus_df)

    raw_cal_metrics, raw_cal_bins = calibration_metrics(probs, y)
    temp_cal_metrics, temp_cal_bins = calibration_metrics(calibrated_probs, y)
    calibration_rows = [
        {
            "dataset": spec.name,
            "split": "test",
            "calibration": "raw",
            "temperature": 1.0,
            **raw_cal_metrics,
        },
        {
            "dataset": spec.name,
            "split": "test",
            "calibration": "temperature_scaled",
            "temperature": float(temperature),
            **temp_cal_metrics,
        },
    ]
    reliability_rows = []
    for mode, bins in (("raw", raw_cal_bins), ("temperature_scaled", temp_cal_bins)):
        for row in bins:
            reliability_rows.append(
                {
                    "dataset": spec.name,
                    "split": "test",
                    "calibration": mode,
                    "temperature": float(temperature if mode == "temperature_scaled" else 1.0),
                    **row,
                }
            )

    if float(np.max(mc_std)) <= 0.0:
        print(f"[WARN] {spec.name}: MC Dropout no produjo variacion positiva.")

    examples = choose_examples(y, probs, mc_std, threshold, max_examples)
    example_rows = []
    importance_rows = []
    faithfulness_rows = []
    counterfactual_rows = []
    aggregate_maps = []
    for item in examples:
        idx = item["index"]
        target_label = int(pred_labels[idx])
        target_confidence = float(target_confidences[idx])
        calibrated_target_confidence = float(calibrated_target_confidences[idx])
        sal_g, sal_l = compute_saliency(model, xg[idx], xl[idx], device, target_label=target_label)
        smooth_g, smooth_l = compute_smoothgrad(
            model,
            xg[idx],
            xl[idx],
            device,
            target_label=target_label,
            samples=smooth_samples,
        )
        ig_g, ig_l = compute_integrated_gradients(
            model,
            xg[idx],
            xl[idx],
            device,
            target_label=target_label,
            steps=ig_steps,
        )
        cam_g, cam_l = compute_gradcam(model, xg[idx], xl[idx], device, target_label=target_label)
        mcam_g, mcam_l = compute_multiscale_gradcam(model, xg[idx], xl[idx], device, target_label=target_label)
        occ_g, occ_l = compute_occlusion_sensitivity(
            model,
            xg[idx],
            xl[idx],
            device,
            target_label=target_label,
        )
        cons_g = consensus_importance(smooth_g, ig_g, occ_g, mcam_g)
        cons_l = consensus_importance(smooth_l, ig_l, occ_l, mcam_l)

        global_lengths = [len(sal_g), len(smooth_g), len(ig_g), len(cam_g), len(mcam_g), len(occ_g), len(cons_g)]
        local_lengths = [len(sal_l), len(smooth_l), len(ig_l), len(cam_l), len(mcam_l), len(occ_l), len(cons_l)]
        if any(v != spec.global_len for v in global_lengths):
            raise RuntimeError(f"Mapa global invalido para {spec.name}: {global_lengths=}")
        if any(v != spec.local_len for v in local_lengths):
            raise RuntimeError(f"Mapa local invalido para {spec.name}: {local_lengths=}")

        fig_name = f"{spec.name}_{item['case']}_idx{idx}_xai_maps.png"
        fig_path = figs_dir / fig_name
        plot_explanation(
            dataset=spec.name,
            case=item["case"],
            idx=idx,
            xg=xg[idx],
            xl=xl[idx],
            sal_g=sal_g,
            sal_l=sal_l,
            smooth_g=smooth_g,
            smooth_l=smooth_l,
            ig_g=ig_g,
            ig_l=ig_l,
            cam_g=cam_g,
            cam_l=cam_l,
            mcam_g=mcam_g,
            mcam_l=mcam_l,
            occ_g=occ_g,
            occ_l=occ_l,
            cons_g=cons_g,
            cons_l=cons_l,
            y_true=int(y[idx]),
            prob=float(probs[idx]),
            threshold=threshold,
            mc_mean=float(mc_mean[idx]),
            mc_std=float(mc_std[idx]),
            out_path=fig_path,
        )
        importance_rows.extend(
            build_importance_metric_rows(
                dataset=spec.name,
                case=item["case"],
                idx=idx,
                sal_g=sal_g,
                sal_l=sal_l,
                smooth_g=smooth_g,
                smooth_l=smooth_l,
                ig_g=ig_g,
                ig_l=ig_l,
                cam_g=cam_g,
                cam_l=cam_l,
                mcam_g=mcam_g,
                mcam_l=mcam_l,
                occ_g=occ_g,
                occ_l=occ_l,
                cons_g=cons_g,
                cons_l=cons_l,
            )
        )
        method_maps = {
            "saliency": (sal_g, sal_l),
            "smoothgrad": (smooth_g, smooth_l),
            "integrated_gradients": (ig_g, ig_l),
            "occlusion": (occ_g, occ_l),
            "gradcam": (cam_g, cam_l),
            "gradcam_multiscale": (mcam_g, mcam_l),
            "consensus": (cons_g, cons_l),
        }
        faithfulness_rows.extend(
            build_deletion_rows(
                model=model,
                dataset=spec.name,
                case=item["case"],
                idx=idx,
                xg=xg[idx],
                xl=xl[idx],
                target_label=target_label,
                method_maps=method_maps,
                device=device,
            )
        )
        counterfactual_rows.extend(
            build_counterfactual_rows(
                model=model,
                dataset=spec.name,
                case=item["case"],
                idx=idx,
                xg=xg[idx],
                xl=xl[idx],
                y_true=int(y[idx]),
                target_label=target_label,
                threshold=threshold,
                device=device,
            )
        )
        aggregate_maps.append(
            {
                "sal_g": sal_g,
                "sal_l": sal_l,
                "smooth_g": smooth_g,
                "smooth_l": smooth_l,
                "ig_g": ig_g,
                "ig_l": ig_l,
                "cam_g": cam_g,
                "cam_l": cam_l,
                "mcam_g": mcam_g,
                "mcam_l": mcam_l,
                "occ_g": occ_g,
                "occ_l": occ_l,
                "cons_g": cons_g,
                "cons_l": cons_l,
            }
        )
        example_rows.append(
            {
                "dataset": spec.name,
                "objective": spec.objective,
                "case": item["case"],
                "index": idx,
                "raw_label": raw_labels[idx],
                "y_true": int(y[idx]),
                "probability": float(probs[idx]),
                "calibrated_probability": float(calibrated_probs[idx]),
                "threshold": float(threshold),
                "calibrated_threshold": float(calibrated_threshold),
                "pred_label": target_label,
                "target_confidence": target_confidence,
                "calibrated_target_confidence": calibrated_target_confidence,
                "mc_mean": float(mc_mean[idx]),
                "mc_std": float(mc_std[idx]),
                "mc_p05": float(mc_p05[idx]),
                "mc_p95": float(mc_p95[idx]),
                "mc_mean_calibrated": float(mc_mean_calibrated[idx]),
                "mc_std_calibrated": float(mc_std_calibrated[idx]),
                "mc_p05_calibrated": float(mc_p05_calibrated[idx]),
                "mc_p95_calibrated": float(mc_p95_calibrated[idx]),
                "figure_path": str(fig_path.relative_to(root)),
            }
        )

    plot_mc_uncertainty(
        dataset=spec.name,
        example_rows=example_rows,
        threshold=threshold,
        out_path=figs_dir / f"{spec.name}_mc_dropout_uncertainty.png",
    )
    plot_mc_scatter(
        dataset=spec.name,
        mc_df=pd.DataFrame(
            {
                "y_true": y.astype(np.int32),
                "probability": probs.astype(np.float32),
                "pred_label": pred_labels.astype(np.int32),
                "mc_std": mc_std.astype(np.float32),
            }
        ),
        threshold=threshold,
        out_path=figs_dir / f"{spec.name}_mc_dropout_scatter.png",
    )
    plot_aggregate_importance(
        dataset=spec.name,
        maps=aggregate_maps,
        out_path=figs_dir / f"{spec.name}_aggregate_importance_selected_examples.png",
    )
    plot_deletion_curves(
        dataset=spec.name,
        rows=faithfulness_rows,
        out_path=figs_dir / f"{spec.name}_faithfulness_deletion_curves.png",
    )
    plot_reliability_diagram(
        dataset=spec.name,
        rows=reliability_rows,
        out_path=figs_dir / f"{spec.name}_reliability_calibration.png",
    )
    plot_counterfactual_effects(
        dataset=spec.name,
        rows=counterfactual_rows,
        out_path=figs_dir / f"{spec.name}_counterfactual_effects.png",
    )
    plot_attention_focus_confusion_matrix(
        dataset=spec.name,
        summary_df=attention_confusion_df,
        out_path=figs_dir / f"{spec.name}_attention_focus_confusion_matrix.png",
    )
    plot_attention_focus_outcome_boxplot(
        dataset=spec.name,
        focus_df=attention_focus_df,
        out_path=figs_dir / f"{spec.name}_attention_focus_outcome_boxplot.png",
    )

    mc_df = pd.DataFrame(
        {
            "dataset": spec.name,
            "index": np.arange(len(y), dtype=np.int32),
            "raw_label": raw_labels,
            "y_true": y.astype(np.int32),
            "probability": probs.astype(np.float32),
            "calibrated_probability": calibrated_probs.astype(np.float32),
            "threshold": float(threshold),
            "calibrated_threshold": float(calibrated_threshold),
            "pred_label": pred_labels.astype(np.int32),
            "target_confidence": target_confidences.astype(np.float32),
            "calibrated_target_confidence": calibrated_target_confidences.astype(np.float32),
            "mc_mean": mc_mean.astype(np.float32),
            "mc_std": mc_std.astype(np.float32),
            "mc_p05": mc_p05.astype(np.float32),
            "mc_p95": mc_p95.astype(np.float32),
            "mc_mean_calibrated": mc_mean_calibrated.astype(np.float32),
            "mc_std_calibrated": mc_std_calibrated.astype(np.float32),
            "mc_p05_calibrated": mc_p05_calibrated.astype(np.float32),
            "mc_p95_calibrated": mc_p95_calibrated.astype(np.float32),
        }
    )
    plot_outcome_uncertainty(
        dataset=spec.name,
        mc_df=mc_df,
        out_path=figs_dir / f"{spec.name}_outcome_uncertainty_summary.png",
    )
    return (
        example_rows,
        mc_df,
        importance_rows,
        faithfulness_rows,
        calibration_rows,
        reliability_rows,
        counterfactual_rows,
        attention_focus_rows,
    )


def write_readme(root: Path, mc_samples: int) -> None:
    readme = root / "resultados_xai" / "README.md"
    text = f"""# Resultados XAI

Esta carpeta contiene explicaciones visuales para los mejores modelos guardados en `mejores_resultados/`.

## Metodos usados

**Saliency Maps**: calcula el gradiente absoluto del logit de la clase explicada respecto a cada punto de la curva de luz. Los puntos mas intensos indican zonas donde pequenos cambios de entrada afectan mas la prediccion.

**SmoothGrad**: promedia varios Saliency Maps calculados sobre versiones levemente perturbadas de la misma curva. Reduce ruido de gradiente y entrega mapas mas estables visualmente.

**Integrated Gradients**: acumula gradientes desde una linea base neutra hasta la curva real. Es menos ruidoso que el gradiente simple y ayuda a confirmar si la relevancia se sostiene al construir la senal completa.

**Grad-CAM 1D**: usa activaciones y gradientes de la ultima convolucion 1D de cada rama del modelo. En esta arquitectura se interpreta como diagnostico secundario, porque el pooling temporal puede generar mapas gruesos o desplazados, especialmente en TESS.

**Grad-CAM multiescala**: combina mapas generados desde varias capas convolucionales. La idea es recuperar evidencia temporal mas fina desde capas tempranas y compararla con el Grad-CAM clasico.

**Occlusion Sensitivity**: enmascara ventanas temporales y mide cuanto cambia la probabilidad del modelo. Es una prueba directa de sensibilidad: si tapar una zona cambia mucho la prediccion, esa zona es importante.

**Mapa de consenso**: combina SmoothGrad, Integrated Gradients, Occlusion y Grad-CAM multiescala. Se usa como resumen visual estable, mientras que la fidelidad final se valida con la prueba de borrado. En los resultados actuales, Occlusion es el metodo mas fuerte por intervencion.

**Fidelidad por borrado**: borra progresivamente los puntos mas importantes segun cada mapa y mide la caida de confianza hacia la clase explicada. Un metodo mas fiel deberia producir una caida mas rapida.

**Contrafactuales simples**: modifican la ventana central del transito, por ejemplo aplanandola, profundizandola o reduciendola, y miden como cambia la probabilidad. Esto ayuda a discutir si la decision depende realmente de la morfologia del transito.

**Conteo de focos de atencion**: calcula Saliency target-aware para todo el test set y cuenta componentes conectados de alta relevancia en las vistas global y local. La idea es medir si los aciertos tienen pocos focos compactos y si los errores presentan atencion mas dispersa.

**Calibracion de probabilidad**: ajusta una temperatura escalar usando el split de validacion. No cambia los pesos del modelo; solo permite reportar probabilidades calibradas, ECE, Brier, NLL y diagramas de confiabilidad.

**MC Dropout**: ejecuta el modelo {mc_samples} veces con dropout activo durante inferencia. La media resume la probabilidad estimada y la desviacion estandar aproxima la incertidumbre predictiva.

## Como interpretar las figuras

Cada figura muestra la vista global y local de una curva de luz. El color de fondo es el mapa de importancia: regiones mas brillantes tienen mayor relevancia para la clase explicada. En ejemplos positivos se explica evidencia a favor de la clase positiva; en ejemplos negativos se explica evidencia a favor de la clase negativa. Esto evita comparar todos los casos contra el mismo logit positivo.

La banda verde marca una ventana central alrededor del transito fase-plegado. No es una regla absoluta, pero sirve como referencia visual para discutir si la relevancia del modelo cae cerca de la region esperada.

## Lectura cuantitativa

Ademas de las figuras individuales, se genera `xai_importance_metrics.csv`. Esta tabla mide, para cada ejemplo y cada vista, el indice del pico de relevancia, la distancia del pico al centro, la fraccion de importancia dentro de la ventana central y el solapamiento entre los puntos mas relevantes de distintos metodos. Tambien se genera `xai_faithfulness_metrics.csv`, que mide cuanto cae la confianza al borrar las regiones mas relevantes.

Las tablas `xai_calibration_summary.csv` y `xai_reliability_bins.csv` cuantifican calibracion antes y despues de temperature scaling. `xai_counterfactual_summary.csv` mide cambios de probabilidad bajo intervenciones simples. `xai_outcome_summary.csv` resume incertidumbre y confianza por TP, TN, FP y FN. Las tablas `xai_attention_focus_*.csv` resumen cuantas zonas de atencion aparecen por muestra y como varia esa cantidad dentro de cada celda de la matriz de confusion.

## Se podran usar estas explicaciones para mejorar el modelo?

Si, pero no de forma directa como si los mapas XAI fueran etiquetas verdaderas. La forma mas segura es usarlas como senales auxiliares de diagnostico. Por ejemplo, si un falso positivo concentra mucha atencion fuera de la ventana central del transito, eso puede indicar ruido, mala normalizacion, un evento secundario o una morfologia que el modelo esta confundiendo. Si un falso negativo muestra atencion muy dispersa o baja centralidad, puede sugerir que el transito quedo mal centrado, que la senal esta degradada o que el preprocesamiento elimino informacion util.

En una etapa futura, estas explicaciones podrian mejorar el desempeno de cuatro maneras. Primero, como auditoria de datos: revisar manualmente muestras con alta incertidumbre, muchos focos dispersos o baja fidelidad para detectar etiquetas dudosas, duplicados o curvas mal preprocesadas. Segundo, como detector de riesgo: entrenar un modelo auxiliar que use `total_focus_count`, entropia de atencion, centralidad, Faithfulness Correlation, MC Dropout y confianza para decidir cuando una prediccion debe aceptarse, revisarse o cambiar de umbral. Tercero, como guia de preprocesamiento: mejorar centrado, suavizado, resampling o ventanas locales cuando los mapas muestran que el modelo mira sistematicamente regiones no fisicas. Cuarto, como regularizacion explicable: penalizar atencion fuera de la zona esperada del transito, siempre que esa restriccion este justificada fisicamente y se valide en un test independiente.

La limitacion principal es que XAI no prueba causalidad por si solo. En redes neuronales no lineales, un mapa puede verse razonable y aun asi no ser completamente fiel al comportamiento interno del modelo. Por eso, cualquier mejora basada en XAI debe validarse con metricas de clasificacion en test, calibracion e idealmente con pruebas de fidelidad como borrado, Occlusion y Faithfulness Correlation. En esta entrega se deja planteado como trabajo futuro: usar las explicaciones no para reemplazar el clasificador, sino para mejorar el control de calidad, seleccionar ejemplos dificiles y construir alertas de predicciones poco confiables.

## Archivos

- `figuras/*_xai_maps.png`: curvas con Saliency, SmoothGrad, Integrated Gradients, Occlusion, Grad-CAM 1D, Grad-CAM multiescala y Consenso.
- `figuras/*_mc_dropout_uncertainty.png`: resumen visual de incertidumbre.
- `figuras/*_mc_dropout_scatter.png`: probabilidad vs incertidumbre para todo el test.
- `figuras/*_aggregate_importance_selected_examples.png`: importancia promedio de los ejemplos seleccionados.
- `figuras/*_faithfulness_deletion_curves.png`: caida de confianza al borrar puntos importantes.
- `figuras/*_reliability_calibration.png`: diagrama de confiabilidad antes y despues de calibracion.
- `figuras/*_counterfactual_effects.png`: efecto promedio de intervenciones contrafactuales.
- `figuras/*_outcome_uncertainty_summary.png`: incertidumbre media por tipo de resultado.
- `figuras/*_attention_focus_confusion_matrix.png`: matriz de confusion normalizada por fila real y anotada con conteos, media y desviacion de focos.
- `figuras/*_attention_focus_outcome_boxplot.png`: distribucion de focos y dispersion temporal por TP/TN/FP/FN.
- `tablas/xai_examples_summary.csv`: ejemplos explicados y sus metricas principales.
- `tablas/mc_dropout_summary.csv`: estadisticas MC Dropout para todo el test set procesado.
- `tablas/xai_importance_metrics.csv`: metricas cuantitativas de concentracion y solapamiento de relevancia.
- `tablas/xai_faithfulness_metrics.csv`: prueba cuantitativa de fidelidad por borrado.
- `tablas/xai_calibration_summary.csv`: ECE, MCE, Brier y NLL antes/despues de calibrar.
- `tablas/xai_reliability_bins.csv`: bins usados para los diagramas de confiabilidad.
- `tablas/xai_counterfactual_summary.csv`: cambios de probabilidad por contrafactual.
- `tablas/xai_outcome_summary.csv`: resumen por TP, TN, FP y FN.
- `tablas/xai_attention_focus_by_sample.csv`: conteo de focos por muestra.
- `tablas/xai_attention_focus_confusion_matrix.csv`: resumen tipo matriz de confusion, incluyendo proporcion normalizada por etiqueta real.
- `tablas/xai_attention_focus_outcome_summary.csv`: resumen agregado por TP/TN/FP/FN.
"""
    readme.write_text(text, encoding="utf-8")


def cleanup_previous_outputs(root: Path, selected: list[str]) -> None:
    figs_dir = root / "resultados_xai" / "figuras"
    tables_dir = root / "resultados_xai" / "tablas"
    if figs_dir.exists():
        for name in selected:
            for path in figs_dir.glob(f"{name}_*_saliency_gradcam.png"):
                path.unlink()
            for path in figs_dir.glob(f"{name}_*_xai_maps.png"):
                path.unlink()
            for suffix in (
                "mc_dropout_uncertainty",
                "mc_dropout_scatter",
                "aggregate_importance_selected_examples",
                "faithfulness_deletion_curves",
                "reliability_calibration",
                "counterfactual_effects",
                "outcome_uncertainty_summary",
                "attention_focus_confusion_matrix",
                "attention_focus_outcome_boxplot",
            ):
                path = figs_dir / f"{name}_{suffix}.png"
                if path.exists():
                    path.unlink()
    if tables_dir.exists():
        for filename in (
            "xai_examples_summary.csv",
            "mc_dropout_summary.csv",
            "xai_importance_metrics.csv",
            "xai_faithfulness_metrics.csv",
            "xai_calibration_summary.csv",
            "xai_reliability_bins.csv",
            "xai_counterfactual_summary.csv",
            "xai_outcome_summary.csv",
            "xai_attention_focus_by_sample.csv",
            "xai_attention_focus_confusion_matrix.csv",
            "xai_attention_focus_outcome_summary.csv",
        ):
            path = tables_dir / filename
            if path.exists():
                path.unlink()


def run_xai(
    root: Path | str = ".",
    dataset: str = "both",
    mc_samples: int = 100,
    max_examples: int = 6,
    batch_size: int = 256,
    smooth_samples: int = 32,
    ig_steps: int = 64,
    device_name: str | None = None,
) -> dict[str, pd.DataFrame]:
    root = Path(root).resolve()
    set_seed(42)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    selected = ["kepler", "tess"] if dataset == "both" else [dataset]

    tables_dir = root / "resultados_xai" / "tablas"
    (root / "resultados_xai" / "figuras").mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    cleanup_previous_outputs(root, selected)

    all_examples: list[dict] = []
    all_mc = []
    all_importance_rows: list[dict] = []
    all_faithfulness_rows: list[dict] = []
    all_calibration_rows: list[dict] = []
    all_reliability_rows: list[dict] = []
    all_counterfactual_rows: list[dict] = []
    all_attention_focus_rows: list[dict] = []
    print(f"Dispositivo: {device}")
    for name in selected:
        spec = DATASETS[name]
        print(
            f"\nProcesando {name.upper()} con {mc_samples} muestras MC Dropout, "
            f"{smooth_samples} muestras SmoothGrad y {ig_steps} pasos IG..."
        )
        (
            rows,
            mc_df,
            importance_rows,
            faithfulness_rows,
            calibration_rows,
            reliability_rows,
            counterfactual_rows,
            attention_focus_rows,
        ) = run_dataset(
            root=root,
            spec=spec,
            device=device,
            mc_samples=mc_samples,
            max_examples=max_examples,
            batch_size=batch_size,
            smooth_samples=smooth_samples,
            ig_steps=ig_steps,
        )
        all_examples.extend(rows)
        all_mc.append(mc_df)
        all_importance_rows.extend(importance_rows)
        all_faithfulness_rows.extend(faithfulness_rows)
        all_calibration_rows.extend(calibration_rows)
        all_reliability_rows.extend(reliability_rows)
        all_counterfactual_rows.extend(counterfactual_rows)
        all_attention_focus_rows.extend(attention_focus_rows)
        print(f"{name.upper()}: ejemplos explicados={len(rows)}, test={len(mc_df)}")

    examples_df = pd.DataFrame(all_examples)
    mc_summary_df = pd.concat(all_mc, ignore_index=True) if all_mc else pd.DataFrame()
    importance_df = pd.DataFrame(all_importance_rows)
    faithfulness_df = pd.DataFrame(all_faithfulness_rows)
    calibration_df = pd.DataFrame(all_calibration_rows)
    reliability_df = pd.DataFrame(all_reliability_rows)
    counterfactual_df = pd.DataFrame(all_counterfactual_rows)
    attention_focus_df = pd.DataFrame(all_attention_focus_rows)
    attention_focus_confusion_df = attention_confusion_summary(attention_focus_df)
    attention_focus_outcome_df = attention_outcome_summary(attention_focus_df)
    if not mc_summary_df.empty:
        mc_summary_df["outcome"] = outcome_labels(
            mc_summary_df["y_true"].to_numpy(),
            mc_summary_df["pred_label"].to_numpy(),
        )
        outcome_df = (
            mc_summary_df.groupby(["dataset", "outcome"], as_index=False)
            .agg(
                n=("index", "count"),
                probability_mean=("probability", "mean"),
                calibrated_probability_mean=("calibrated_probability", "mean"),
                target_confidence_mean=("target_confidence", "mean"),
                calibrated_target_confidence_mean=("calibrated_target_confidence", "mean"),
                mc_std_mean=("mc_std", "mean"),
                mc_std_calibrated_mean=("mc_std_calibrated", "mean"),
            )
            .sort_values(["dataset", "outcome"])
        )
    else:
        outcome_df = pd.DataFrame()
    examples_df.to_csv(tables_dir / "xai_examples_summary.csv", index=False)
    mc_summary_df.to_csv(tables_dir / "mc_dropout_summary.csv", index=False)
    importance_df.to_csv(tables_dir / "xai_importance_metrics.csv", index=False)
    faithfulness_df.to_csv(tables_dir / "xai_faithfulness_metrics.csv", index=False)
    calibration_df.to_csv(tables_dir / "xai_calibration_summary.csv", index=False)
    reliability_df.to_csv(tables_dir / "xai_reliability_bins.csv", index=False)
    counterfactual_df.to_csv(tables_dir / "xai_counterfactual_summary.csv", index=False)
    outcome_df.to_csv(tables_dir / "xai_outcome_summary.csv", index=False)
    attention_focus_df.to_csv(tables_dir / "xai_attention_focus_by_sample.csv", index=False)
    attention_focus_confusion_df.to_csv(tables_dir / "xai_attention_focus_confusion_matrix.csv", index=False)
    attention_focus_outcome_df.to_csv(tables_dir / "xai_attention_focus_outcome_summary.csv", index=False)
    write_readme(root, mc_samples)
    print(f"\nListo. Resultados guardados en: {root / 'resultados_xai'}")
    return {
        "examples": examples_df,
        "mc": mc_summary_df,
        "importance": importance_df,
        "faithfulness": faithfulness_df,
        "calibration": calibration_df,
        "reliability": reliability_df,
        "counterfactual": counterfactual_df,
        "outcome": outcome_df,
        "attention_focus": attention_focus_df,
        "attention_focus_confusion": attention_focus_confusion_df,
        "attention_focus_outcome": attention_focus_outcome_df,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera Saliency, SmoothGrad, Integrated Gradients, Grad-CAM 1D, "
            "Grad-CAM multiescala, Occlusion, consenso, calibracion, "
            "contrafactuales y MC Dropout para los mejores modelos."
        )
    )
    parser.add_argument("--dataset", choices=["both", "kepler", "tess"], default="both")
    parser.add_argument("--mc-samples", type=int, default=100)
    parser.add_argument("--max-examples", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--smooth-samples", type=int, default=32)
    parser.add_argument("--ig-steps", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Raiz del proyecto. Por defecto usa la carpeta donde esta este script.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mc_samples < 2:
        raise ValueError("--mc-samples debe ser >= 2 para estimar incertidumbre.")
    if args.max_examples < 1:
        raise ValueError("--max-examples debe ser >= 1.")
    if args.smooth_samples < 1:
        raise ValueError("--smooth-samples debe ser >= 1.")
    if args.ig_steps < 2:
        raise ValueError("--ig-steps debe ser >= 2.")
    run_xai(
        root=args.root,
        dataset=args.dataset,
        mc_samples=args.mc_samples,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        smooth_samples=args.smooth_samples,
        ig_steps=args.ig_steps,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
