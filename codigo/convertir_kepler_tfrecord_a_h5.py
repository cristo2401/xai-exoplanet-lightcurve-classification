from __future__ import annotations

import argparse
import glob
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf


def _to_label(training_set: str) -> int:
    # Configuracion usada en este proyecto:
    # positivo = PC, negativo = todo lo demas.
    return 1 if training_set == "PC" else 0


def load_split(files: list[str]):
    xg, xl, y, training_set_raw, pred_class_raw, kepid = [], [], [], [], [], []
    for rec in tf.data.TFRecordDataset(files):
        ex = tf.train.Example()
        ex.ParseFromString(rec.numpy())
        f = ex.features.feature

        ts = f["av_training_set"].bytes_list.value[0].decode()
        pc = f["av_pred_class"].bytes_list.value[0].decode()
        g = np.array(f["global_view"].float_list.value, dtype=np.float32)
        l = np.array(f["local_view"].float_list.value, dtype=np.float32)
        kid = int(f["kepid"].int64_list.value[0])

        xg.append(g)
        xl.append(l)
        y.append(_to_label(ts))
        training_set_raw.append(ts.encode("utf-8"))
        pred_class_raw.append(pc.encode("utf-8"))
        kepid.append(kid)

    return (
        np.stack(xg, axis=0).astype(np.float32),
        np.stack(xl, axis=0).astype(np.float32),
        np.array(y, dtype=np.int32),
        np.array(training_set_raw, dtype="S8"),
        np.array(pred_class_raw, dtype="S8"),
        np.array(kepid, dtype=np.int64),
    )


def save_h5(output_path: Path, data: dict[str, tuple]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as hf:
        for split, (xg, xl, y, ts_raw, pc_raw, kid) in data.items():
            grp = hf.create_group(split)
            grp.create_dataset("global_view", data=xg, compression="gzip")
            grp.create_dataset("local_view", data=xl, compression="gzip")
            grp.create_dataset("labels", data=y, compression="gzip")
            grp.create_dataset("av_training_set_raw", data=ts_raw, compression="gzip")
            grp.create_dataset("av_pred_class_raw", data=pc_raw, compression="gzip")
            grp.create_dataset("kepid", data=kid, compression="gzip")


def main():
    parser = argparse.ArgumentParser(
        description="Convierte TFRecords de Kepler a HDF5 (train/val/test)."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tfrecords_KEPLER",
        help="Carpeta con TFRecords de Kepler.",
    )
    parser.add_argument(
        "--output_h5",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datos_procesados_h5" / "kepler_dataset.h5",
        help="Ruta del H5 de salida.",
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
        xg, xl, y, ts_raw, pc_raw, kid = load_split(files)
        data[split] = (xg, xl, y, ts_raw, pc_raw, kid)
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
