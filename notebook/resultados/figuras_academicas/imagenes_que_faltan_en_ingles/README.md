# Missing English Figures for Overleaf

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
