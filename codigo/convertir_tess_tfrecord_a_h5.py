from __future__ import annotations

import argparse
import glob
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf


def _to_label(disposition: str, mode: str) -> int:
    if mode == "pc_only":
        return 1 if disposition == "PC" else 0
    if mode == "triage":
        return 1 if disposition in {"PC", "EB"} else 0
    raise ValueError(f"Modo no soportado: {mode}")


def load_split(files: list[str], mode: str):
    xg, xl, y, disp = [], [], [], []
    for rec in tf.data.TFRecordDataset(files):
        ex = tf.train.Example()
        ex.ParseFromString(rec.numpy())
        f = ex.features.feature
        d = f["Disposition"].bytes_list.value[0].decode()
        g = np.array(f["global_view"].float_list.value, dtype=np.float32)
        l = np.array(f["local_view"].float_list.value, dtype=np.float32)
        xg.append(g)
        xl.append(l)
        y.append(_to_label(d, mode))
        disp.append(d.encode("utf-8"))
    return (
        np.stack(xg, axis=0).astype(np.float32),
        np.stack(xl, axis=0).astype(np.float32),
        np.array(y, dtype=np.int32),
        np.array(disp, dtype="S8"),
    )


def save_h5(output_path: Path, data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as hf:
        for split, (xg, xl, y, disp) in data.items():
            grp = hf.create_group(split)
            grp.create_dataset("global_view", data=xg, compression="gzip")
            grp.create_dataset("local_view", data=xl, compression="gzip")
            grp.create_dataset("labels", data=y, compression="gzip")
            grp.create_dataset("disposition_raw", data=disp, compression="gzip")


def main():
    parser = argparse.ArgumentParser(
        description="Convierte TFRecords de TESS a HDF5 con splits train/val/test."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tfrecords_TESS",
        help="Carpeta con TFRecords TESS.",
    )
    parser.add_argument(
        "--output_h5",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datos_procesados_h5" / "tess_dataset_from_tfrecord.h5",
        help="Ruta de salida H5.",
    )
    parser.add_argument(
        "--mode",
        choices=["pc_only", "triage"],
        default="pc_only",
        help="Definición de clase positiva.",
    )
    args = parser.parse_args()

    splits = {
        "train": sorted(glob.glob(str(args.input_dir / "train-*"))),
        "val": sorted(glob.glob(str(args.input_dir / "val-*"))),
        "test": sorted(glob.glob(str(args.input_dir / "test-*"))),
    }

    for s, files in splits.items():
        if not files:
            raise FileNotFoundError(f"No se encontraron archivos para split '{s}' en {args.input_dir}")

    data = {}
    for split, files in splits.items():
        xg, xl, y, disp = load_split(files, args.mode)
        data[split] = (xg, xl, y, disp)
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        print(
            f"[{split}] n={len(y)} pos={pos} neg={neg} "
            f"global_len={xg.shape[1]} local_len={xl.shape[1]}"
        )

    save_h5(args.output_h5, data)
    print(f"\nListo. Archivo generado: {args.output_h5}")


if __name__ == "__main__":
    main()
