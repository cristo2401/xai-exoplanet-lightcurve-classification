import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import average_precision_score, matthews_corrcoef, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


PROJECT_DIR = Path(__file__).resolve().parents[1]
H5_PATH = PROJECT_DIR / "datos_procesados_h5/kepler_dataset.h5"
OUT_DIR = Path("resultados_kepler_hsearch")
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ExpConfig:
    name: str
    preprocess: str = "none"  # none|zscore|robust
    loss: str = "bce"  # bce|focal
    lr: float = 3e-4
    weight_decay: float = 1e-5
    dropout: float = 0.0
    batch_size: int = 128
    epochs: int = 34
    patience: int = 8
    use_sampler: bool = False
    use_pos_weight: bool = False
    flip_prob: float = 0.0
    noise_std: float = 0.0
    seed: int = 42


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
        self.xg = np.expand_dims(xg.astype(np.float32), 1)
        self.xl = np.expand_dims(xl.astype(np.float32), 1)
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
    def __init__(self, gl=2001, ll=201, drop=0.0):
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


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = targets * p + (1.0 - targets) * (1.0 - p)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal = alpha_t * ((1.0 - pt) ** self.gamma) * bce
        return focal.mean()


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
    acc = float((pred == y).mean())
    return {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "mcc": float(mcc),
        "pred_pos_rate": float(pred.mean()),
    }


def make_train_loader(xg, xl, y, cfg: ExpConfig):
    ds = CurveDS(xg, xl, y, flip_prob=cfg.flip_prob, noise_std=cfg.noise_std)
    if cfg.use_sampler:
        cls = np.array([(y == 0).sum(), (y == 1).sum()], dtype=np.float32)
        w = 1.0 / np.maximum(cls, 1.0)
        sw = w[y]
        sampler = WeightedRandomSampler(torch.tensor(sw, dtype=torch.double), len(sw), replacement=True)
        return DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=0)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)


def train_one(cfg: ExpConfig, raw):
    set_seed(cfg.seed)

    xg_tr, xl_tr, y_tr = raw["train"]
    xg_va, xl_va, y_va = raw["val"]
    xg_te, xl_te, y_te = raw["test"]

    xg_tr = preprocess_views(xg_tr, cfg.preprocess)
    xl_tr = preprocess_views(xl_tr, cfg.preprocess)
    xg_va = preprocess_views(xg_va, cfg.preprocess)
    xl_va = preprocess_views(xl_va, cfg.preprocess)
    xg_te = preprocess_views(xg_te, cfg.preprocess)
    xl_te = preprocess_views(xl_te, cfg.preprocess)

    tr_loader = make_train_loader(xg_tr, xl_tr, y_tr, cfg)
    va_loader = DataLoader(CurveDS(xg_va, xl_va, y_va), batch_size=256, shuffle=False, num_workers=0)
    te_loader = DataLoader(CurveDS(xg_te, xl_te, y_te), batch_size=256, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(gl=xg_tr.shape[1], ll=xl_tr.shape[1], drop=cfg.dropout).to(device)

    if cfg.loss == "focal":
        criterion = FocalLoss(gamma=2.0, alpha=0.25)
    elif cfg.loss == "bce":
        if cfg.use_pos_weight:
            pos = max((y_tr == 1).sum(), 1)
            neg = max((y_tr == 0).sum(), 1)
            pw = torch.tensor([neg / pos], dtype=torch.float32, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(cfg.loss)

    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = optim.lr_scheduler.StepLR(opt, step_size=max(10, cfg.epochs // 2), gamma=0.5)

    best_state = None
    best_score = (-1.0, -1.0)
    best_thr = 0.5
    bad = 0

    for _ in range(cfg.epochs):
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
            if bad >= cfg.patience:
                break

    model.load_state_dict(best_state)

    yt, pt = eval_probs(model, te_loader, device)
    m05 = metrics(yt, pt, 0.5)
    mb = metrics(yt, pt, best_thr)
    ap = float(average_precision_score(yt, pt))

    return {
        "cfg": asdict(cfg),
        "train_size": int(len(y_tr)),
        "val_size": int(len(y_va)),
        "test_size": int(len(y_te)),
        "train_pos": int((y_tr == 1).sum()),
        "val_pos": int((y_va == 1).sum()),
        "test_pos": int((y_te == 1).sum()),
        "best_val_f1_0_5": float(best_score[0]),
        "best_val_f1_any": float(best_score[1]),
        "best_thr": float(best_thr),
        "test_0_5": m05,
        "test_best": mb,
        "test_ap": ap,
    }


def audit_dataset(raw):
    (xg_tr, xl_tr, y_tr) = raw["train"]
    (xg_va, xl_va, y_va) = raw["val"]
    (xg_te, xl_te, y_te) = raw["test"]

    def hash_set(xg, xl):
        hs = []
        for i in range(len(xg)):
            hs.append(hashlib.sha1(np.concatenate([xg[i], xl[i]]).astype(np.float32).tobytes()).hexdigest())
        return set(hs)

    htr = hash_set(xg_tr, xl_tr)
    hva = hash_set(xg_va, xl_va)
    hte = hash_set(xg_te, xl_te)

    report = {
        "dataset": "Kepler",
        "source_h5": str(H5_PATH),
        "paper_counts": {"pos": 3094, "neg": 10417, "total": 13511},
        "local_counts": {
            "train": {"n": int(len(y_tr)), "pos": int((y_tr == 1).sum()), "neg": int((y_tr == 0).sum())},
            "val": {"n": int(len(y_va)), "pos": int((y_va == 1).sum()), "neg": int((y_va == 0).sum())},
            "test": {"n": int(len(y_te)), "pos": int((y_te == 1).sum()), "neg": int((y_te == 0).sum())},
        },
        "local_total": {
            "n": int(len(y_tr) + len(y_va) + len(y_te)),
            "pos": int((y_tr == 1).sum() + (y_va == 1).sum() + (y_te == 1).sum()),
            "neg": int((y_tr == 0).sum() + (y_va == 0).sum() + (y_te == 0).sum()),
        },
        "duplicates_within_split": {
            "train": int(len(y_tr) - len(htr)),
            "val": int(len(y_va) - len(hva)),
            "test": int(len(y_te) - len(hte)),
        },
        "cross_split_overlap": {
            "train_val": int(len(htr & hva)),
            "train_test": int(len(htr & hte)),
            "val_test": int(len(hva & hte)),
        },
    }

    with open(OUT_DIR / "kepler_dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    paper_vs_local = pd.DataFrame(
        [
            {
                "source": "Paper Xie 2025 (Table 1)",
                "PC": 3094,
                "Non-PC": 10417,
                "Total": 13511,
                "notes": "Kepler counts reported in paper",
            },
            {
                "source": "Local kepler_dataset.h5 total",
                "PC": report["local_total"]["pos"],
                "Non-PC": report["local_total"]["neg"],
                "Total": report["local_total"]["n"],
                "notes": "train+val+test from local H5",
            },
        ]
    )
    paper_vs_local.to_csv(OUT_DIR / "paper_vs_local_kepler_counts.csv", index=False)


def load_raw_splits():
    with h5py.File(H5_PATH, "r") as hf:
        raw = {
            "train": (
                hf["train"]["global_view"][:].astype(np.float32),
                hf["train"]["local_view"][:].astype(np.float32),
                hf["train"]["labels"][:].astype(np.int32),
            ),
            "val": (
                hf["val"]["global_view"][:].astype(np.float32),
                hf["val"]["local_view"][:].astype(np.float32),
                hf["val"]["labels"][:].astype(np.int32),
            ),
            "test": (
                hf["test"]["global_view"][:].astype(np.float32),
                hf["test"]["local_view"][:].astype(np.float32),
                hf["test"]["labels"][:].astype(np.int32),
            ),
        }
    return raw


def main():
    t0 = time.time()
    raw = load_raw_splits()
    audit_dataset(raw)

    grid = [
        ExpConfig(name="best_effort_repro", preprocess="none", loss="bce", lr=3e-4, dropout=0.0, use_sampler=False, use_pos_weight=False, flip_prob=0.0, noise_std=0.0),
        ExpConfig(name="best_effort_flip", preprocess="none", loss="bce", lr=3e-4, dropout=0.0, use_sampler=False, use_pos_weight=False, flip_prob=0.5, noise_std=0.0),
        ExpConfig(name="best_effort_flip_noise", preprocess="none", loss="bce", lr=3e-4, dropout=0.0, use_sampler=False, use_pos_weight=False, flip_prob=0.5, noise_std=0.005),
        ExpConfig(name="none_bce_posw", preprocess="none", loss="bce", lr=3e-4, dropout=0.0, use_sampler=False, use_pos_weight=True, flip_prob=0.0, noise_std=0.0),
        ExpConfig(name="none_bce_sampler", preprocess="none", loss="bce", lr=3e-4, dropout=0.0, use_sampler=True, use_pos_weight=False, flip_prob=0.0, noise_std=0.0),
        ExpConfig(name="none_bce_sampler_posw", preprocess="none", loss="bce", lr=2e-4, dropout=0.1, use_sampler=True, use_pos_weight=True, flip_prob=0.3, noise_std=0.005),
        ExpConfig(name="robust_bce", preprocess="robust", loss="bce", lr=2e-4, dropout=0.1, use_sampler=False, use_pos_weight=False, flip_prob=0.3, noise_std=0.005),
        ExpConfig(name="zscore_bce", preprocess="zscore", loss="bce", lr=2e-4, dropout=0.1, use_sampler=False, use_pos_weight=False, flip_prob=0.3, noise_std=0.005),
        ExpConfig(name="none_focal", preprocess="none", loss="focal", lr=2e-4, dropout=0.1, use_sampler=False, use_pos_weight=False, flip_prob=0.3, noise_std=0.005),
        ExpConfig(name="robust_focal", preprocess="robust", loss="focal", lr=2e-4, dropout=0.1, use_sampler=False, use_pos_weight=False, flip_prob=0.3, noise_std=0.005),
        ExpConfig(name="none_bce_lr1e4", preprocess="none", loss="bce", lr=1e-4, dropout=0.2, use_sampler=False, use_pos_weight=False, flip_prob=0.3, noise_std=0.005),
        ExpConfig(name="none_bce_lr2e4_drop02", preprocess="none", loss="bce", lr=2e-4, dropout=0.2, use_sampler=False, use_pos_weight=False, flip_prob=0.15, noise_std=0.005),
    ]

    runs = []
    for i, cfg in enumerate(grid, start=1):
        print(f"\\n[{i}/{len(grid)}] {cfg.name}")
        r = train_one(cfg, raw)
        runs.append(r)
        print(
            f"{cfg.name}: F1@0.5={r['test_0_5']['f1']:.4f} MCC@0.5={r['test_0_5']['mcc']:.4f} | "
            f"F1@best={r['test_best']['f1']:.4f} MCC@best={r['test_best']['mcc']:.4f} AP={r['test_ap']:.4f}"
        )

    with open(OUT_DIR / "kepler_hsearch_report.json", "w", encoding="utf-8") as f:
        json.dump({"elapsed_min": (time.time() - t0) / 60.0, "runs": runs}, f, indent=2, ensure_ascii=False)

    rows = []
    for r in runs:
        c = r["cfg"]
        rows.append(
            {
                "name": c["name"],
                "preprocess": c["preprocess"],
                "loss": c["loss"],
                "lr": c["lr"],
                "dropout": c["dropout"],
                "use_sampler": c["use_sampler"],
                "use_pos_weight": c["use_pos_weight"],
                "flip_prob": c["flip_prob"],
                "noise_std": c["noise_std"],
                "f1@0.5": r["test_0_5"]["f1"],
                "mcc@0.5": r["test_0_5"]["mcc"],
                "best_thr": r["best_thr"],
                "f1@best": r["test_best"]["f1"],
                "mcc@best": r["test_best"]["mcc"],
                "ap": r["test_ap"],
            }
        )

    df = pd.DataFrame(rows).sort_values(["f1@best", "mcc@best", "ap"], ascending=False)
    df.to_csv(OUT_DIR / "kepler_hsearch_summary.csv", index=False)

    # compact comparison with paper
    best = df.iloc[0].to_dict()
    paper = {"acc": 0.962, "precision": 0.89, "recall": 0.95, "f1": 0.957, "mcc": 0.894}
    comp = pd.DataFrame(
        [
            {
                "source": "paper_xie_2025_kepler",
                "accuracy": paper["acc"],
                "precision": paper["precision"],
                "recall": paper["recall"],
                "f1": paper["f1"],
                "mcc": paper["mcc"],
                "ap": np.nan,
            },
            {
                "source": f"best_local_{best['name']}",
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": best["f1@best"],
                "mcc": best["mcc@best"],
                "ap": best["ap"],
            },
        ]
    )
    comp.to_csv(OUT_DIR / "paper_vs_best_kepler_hsearch.csv", index=False)

    print("\\nTop configs:")
    print(df.head(10).to_string(index=False))
    print("\\nSaved:", (OUT_DIR / "kepler_hsearch_summary.csv").resolve())


if __name__ == "__main__":
    main()
