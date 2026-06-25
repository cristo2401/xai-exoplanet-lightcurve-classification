import glob
import hashlib
import itertools
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import average_precision_score, matthews_corrcoef, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


PROJECT_DIR = Path(__file__).resolve().parents[1]
BASE = PROJECT_DIR / "tfrecords_TESS"
OUT = Path("resultados_tess_triage_hsearch")
OUT.mkdir(parents=True, exist_ok=True)


@dataclass
class Cfg:
    name: str
    seed: int = 42
    flip_prob: float = 0.0
    noise_std: float = 0.0
    lr: float = 2e-4
    weight_decay: float = 1e-5
    dropout: float = 0.2
    batch_size: int = 128
    epochs: int = 40
    patience: int = 10
    use_sampler: bool = True
    use_pos_weight: bool = True


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def robust_preprocess(x: np.ndarray) -> np.ndarray:
    med = np.median(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True) + 1e-6
    x = (x - med) / std
    return np.clip(x, -5.0, 5.0).astype(np.float32)


def load_rows(files):
    rows = []
    for rec in tf.data.TFRecordDataset(files):
        ex = tf.train.Example()
        ex.ParseFromString(rec.numpy())
        f = ex.features.feature
        disp = f["Disposition"].bytes_list.value[0].decode()
        g = np.array(f["global_view"].float_list.value, dtype=np.float32)
        l = np.array(f["local_view"].float_list.value, dtype=np.float32)
        h = hashlib.sha1(np.concatenate([g, l]).tobytes()).hexdigest()
        rows.append({"h": h, "disp": disp, "g": g, "l": l})
    seen = set()
    dedup = []
    for r in rows:
        if r["h"] in seen:
            continue
        seen.add(r["h"])
        dedup.append(r)
    return dedup


def triage_arrays(rows):
    xg = np.stack([r["g"] for r in rows], axis=0).astype(np.float32)
    xl = np.stack([r["l"] for r in rows], axis=0).astype(np.float32)
    y = np.array([1 if r["disp"] in {"PC", "EB"} else 0 for r in rows], dtype=np.int32)
    return xg, xl, y


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
    def __init__(self, gl=201, ll=61, drop=0.2):
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
    }


def make_train_loader(xg, xl, y, cfg: Cfg):
    ds = CurveDS(xg, xl, y, flip_prob=cfg.flip_prob, noise_std=cfg.noise_std)
    if cfg.use_sampler:
        cls = np.array([(y == 0).sum(), (y == 1).sum()], dtype=np.float32)
        w = 1.0 / np.maximum(cls, 1.0)
        sw = w[y]
        sampler = WeightedRandomSampler(torch.tensor(sw, dtype=torch.double), len(sw), replacement=True)
        return DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=0)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)


def run_cfg(cfg: Cfg, data):
    set_seed(cfg.seed)
    xg_tr, xl_tr, y_tr, xg_va, xl_va, y_va, xg_te, xl_te, y_te = data

    tr_loader = make_train_loader(xg_tr, xl_tr, y_tr, cfg)
    va_loader = DataLoader(CurveDS(xg_va, xl_va, y_va), batch_size=256, shuffle=False, num_workers=0)
    te_loader = DataLoader(CurveDS(xg_te, xl_te, y_te), batch_size=256, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(gl=201, ll=61, drop=cfg.dropout).to(device)

    if cfg.use_pos_weight:
        pos = max((y_tr == 1).sum(), 1)
        neg = max((y_tr == 0).sum(), 1)
        pw = torch.tensor([neg / pos], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        criterion = nn.BCEWithLogitsLoss()

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
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "n_test": int(len(y_te)),
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


def main():
    t0 = time.time()

    train_rows = load_rows(sorted(glob.glob(str(BASE / "train-*"))))
    val_rows = load_rows(sorted(glob.glob(str(BASE / "val-*"))))
    test_rows = load_rows(sorted(glob.glob(str(BASE / "test-*"))))

    xg_tr, xl_tr, y_tr = triage_arrays(train_rows)
    xg_va, xl_va, y_va = triage_arrays(val_rows)
    xg_te, xl_te, y_te = triage_arrays(test_rows)

    xg_tr, xl_tr = robust_preprocess(xg_tr), robust_preprocess(xl_tr)
    xg_va, xl_va = robust_preprocess(xg_va), robust_preprocess(xl_va)
    xg_te, xl_te = robust_preprocess(xg_te), robust_preprocess(xl_te)

    data = (xg_tr, xl_tr, y_tr, xg_va, xl_va, y_va, xg_te, xl_te, y_te)

    grid = [
        Cfg(name="base", flip_prob=0.5, noise_std=0.01, lr=2e-4, dropout=0.2, use_sampler=True, use_pos_weight=True),
        Cfg(name="no_posw", flip_prob=0.5, noise_std=0.01, lr=2e-4, dropout=0.2, use_sampler=True, use_pos_weight=False),
        Cfg(name="no_sampler", flip_prob=0.5, noise_std=0.01, lr=2e-4, dropout=0.2, use_sampler=False, use_pos_weight=True),
        Cfg(name="no_sampler_no_posw", flip_prob=0.5, noise_std=0.01, lr=2e-4, dropout=0.2, use_sampler=False, use_pos_weight=False),
        Cfg(name="no_aug_sampler_posw", flip_prob=0.0, noise_std=0.0, lr=2e-4, dropout=0.2, use_sampler=True, use_pos_weight=True),
        Cfg(name="no_aug_sampler_no_posw", flip_prob=0.0, noise_std=0.0, lr=2e-4, dropout=0.2, use_sampler=True, use_pos_weight=False),
        Cfg(name="lr1e4_sampler_posw", flip_prob=0.5, noise_std=0.01, lr=1e-4, dropout=0.2, use_sampler=True, use_pos_weight=True),
        Cfg(name="drop01_sampler_posw", flip_prob=0.5, noise_std=0.01, lr=2e-4, dropout=0.1, use_sampler=True, use_pos_weight=True),
        Cfg(name="drop03_sampler_posw", flip_prob=0.5, noise_std=0.01, lr=2e-4, dropout=0.3, use_sampler=True, use_pos_weight=True),
        Cfg(name="gentle_aug_no_sampler", flip_prob=0.15, noise_std=0.005, lr=1.5e-4, dropout=0.2, use_sampler=False, use_pos_weight=True),
    ]

    runs = []
    for i, cfg in enumerate(grid, start=1):
        print(f"\\n[{i}/{len(grid)}] {cfg.name}")
        r = run_cfg(cfg, data)
        runs.append(r)
        print(
            f"{cfg.name}: F1@0.5={r['test_0_5']['f1']:.4f} MCC@0.5={r['test_0_5']['mcc']:.4f} | "
            f"F1@best={r['test_best']['f1']:.4f} MCC@best={r['test_best']['mcc']:.4f} AP={r['test_ap']:.4f}"
        )

    with open(OUT / "hsearch_report.json", "w", encoding="utf-8") as f:
        json.dump({"elapsed_min": (time.time() - t0) / 60.0, "runs": runs}, f, indent=2, ensure_ascii=False)

    rows = []
    for r in runs:
        c = r["cfg"]
        rows.append({
            "name": c["name"],
            "flip_prob": c["flip_prob"],
            "noise_std": c["noise_std"],
            "lr": c["lr"],
            "dropout": c["dropout"],
            "use_sampler": c["use_sampler"],
            "use_pos_weight": c["use_pos_weight"],
            "f1@0.5": r["test_0_5"]["f1"],
            "mcc@0.5": r["test_0_5"]["mcc"],
            "best_thr": r["best_thr"],
            "f1@best": r["test_best"]["f1"],
            "mcc@best": r["test_best"]["mcc"],
            "ap": r["test_ap"],
        })

    df = pd.DataFrame(rows).sort_values(["f1@best", "mcc@best", "ap"], ascending=False)
    df.to_csv(OUT / "hsearch_summary.csv", index=False)

    print("\\nTop configs:")
    print(df.head(10).to_string(index=False))
    print("\\nSaved:", (OUT / "hsearch_summary.csv").resolve())


if __name__ == "__main__":
    main()
