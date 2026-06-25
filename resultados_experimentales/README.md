# Entrega 2: Resultados Experimentales (Preliminares)

## 1. Objetivo de esta entrega

Este documento resume resultados **preliminares** del sistema de clasificacion de curvas de luz de exoplanetas en Kepler y TESS. Se incluyen primeras pruebas, evolucion experimental, comparacion contra el paper de referencia y una descripcion del hardware/software utilizado. La fecha de consolidacion de esta entrega es **27 de mayo de 2026**.

## 2. Estructura de la carpeta

- `tablas/tabla_01_evolucion_preliminar.csv`: evolucion de metricas por etapa experimental.
- `tablas/tabla_02_paper_vs_mejor_local.csv`: comparacion paper vs mejor resultado local.
- `tablas/tabla_03_auditoria_datos.csv`: comparacion de volumen de datos y solapamientos detectados.
- `tablas/tabla_04_hw_sw.csv`: descripcion del entorno de ejecucion.
- `figuras/fig_01_paper_vs_local.png`: comparacion visual paper vs mejor local.
- `figuras/fig_02_evolucion_f1_mcc.png`: evolucion de F1 y MCC por etapa.
- `figuras/fig_03_top_configs_kepler.png`: top configuraciones en barrido de Kepler.
- `figuras/fig_04_top_configs_tess.png`: top configuraciones en barrido de TESS.
- `figuras/fig_05_auditoria_datos.png`: resumen visual de auditoria de datos.
- `figuras/ref_*`: curvas PR de referencia de corridas iniciales/estrictas.
- `generar_resultados_experimentales.py`: script para regenerar tablas y figuras.

## 3. Hardware y software utilizados

Resumen del entorno usado para las pruebas preliminares:

- SO: Ubuntu 24.04.2 LTS
- Kernel: 6.17.0-29-generic
- CPU: Intel Core i9-10900F (20 hilos logicos)
- GPU: NVIDIA GeForce RTX 3070 (8 GB VRAM)
- Driver/CUDA reportado por `nvidia-smi`: Driver 580.95.05, CUDA 13.0
- Entorno Python: Python 3.12.3, PyTorch 2.5.1, TensorFlow 2.18.1, NumPy 2.0.1, pandas 2.3.1, scikit-learn 1.7.1

Version tabulada completa en `tablas/tabla_04_hw_sw.csv`.

## 4. Resultados preliminares: primeras pruebas y evolucion

Para esta entrega se consolidaron cuatro etapas principales por dataset:

1. `baseline_0.5`: corrida inicial de replica.
2. `strict_0.5`: replica estricta buscando acercar el protocolo del paper.
3. `best_effort_0.5`: ajuste preliminar de hiperparametros.
4. `mejor_local_0.5`: mejor modelo local encontrado hasta ahora.
5. `mejor_local_best_thr`: mismo mejor modelo, evaluado al umbral optimo segun validacion.

### 4.1 Kepler (objetivo PC vs no-PC)

| Etapa | Acc | Prec | Recall | F1 | MCC | AP |
|---|---:|---:|---:|---:|---:|---:|
| Baseline @0.5 | 0.919 | 0.797 | 0.864 | 0.829 | 0.777 | 0.888 |
| Strict @0.5 | 0.902 | 0.809 | 0.752 | 0.779 | 0.717 | 0.737 |
| Best effort @0.5 | 0.930 | 0.823 | 0.887 | 0.854 | 0.809 | 0.916 |
| Mejor local @0.5 | 0.967 | 0.907 | 0.953 | 0.930 | 0.908 | 0.945 |
| Mejor local @thr* | 0.967 | 0.907 | 0.953 | 0.930 | 0.908 | 0.945 |

Lectura preliminar: Kepler mejora de forma monotona desde baseline hacia el mejor local, especialmente en F1 y MCC, lo que sugiere una separacion de clases mas robusta despues del ajuste de preprocesamiento y regularizacion.

### 4.2 TESS

| Etapa | Objetivo | Acc | Prec | Recall | F1 | MCC | AP |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline @0.5 | segun baseline | 0.970 | 0.000 | 0.000 | 0.000 | 0.000 | 0.361 |
| Strict @0.5 | PC vs no-PC | 0.973 | 1.000 | 0.077 | 0.143 | 0.274 | 0.431 |
| Best effort @0.5 | segun best effort | 0.970 | 0.000 | 0.000 | 0.000 | 0.000 | 0.123 |
| Mejor local @0.5 | (PC+EB) vs resto | 0.961 | 0.841 | 0.950 | 0.892 | 0.871 | 0.975 |
| Mejor local @thr* | (PC+EB) vs resto | 0.971 | 0.902 | 0.928 | 0.915 | 0.898 | 0.975 |

Lectura preliminar: en TESS, las primeras pruebas en formulacion estricta `PC vs resto` mostraron bajo F1/MCC a pesar de accuracy alta, evidenciando el efecto del desbalance y de la definicion de objetivo. La formulacion triage `(PC+EB) vs resto` entrega mejoras sustantivas en F1, MCC y AP.

## 5. Comparacion con el paper

Tabla resumen (threshold 0.5):

| Dataset | Fuente | Objetivo | Acc | Prec | Recall | F1 | MCC | AP |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Kepler | Paper | PC vs no-PC | 0.962 | 0.890 | 0.950 | 0.957 | 0.894 | - |
| Kepler | Mejor local | PC vs no-PC | 0.967 | 0.907 | 0.953 | 0.930 | 0.908 | 0.945 |
| TESS | Paper | PC vs no-PC | 0.999 | 0.970 | 1.000 | 0.995 | 0.979 | - |
| TESS | Mejor local | (PC+EB) vs resto | 0.961 | 0.841 | 0.950 | 0.892 | 0.871 | 0.975 |

Interpretacion preliminar:

- En **Kepler**, el mejor local supera al paper en Accuracy, Precision, Recall y MCC, pero queda por debajo en F1.
- En **TESS**, la comparacion no es estrictamente equivalente en el mejor local de esta entrega porque el objetivo es triage y no `PC vs no-PC`.
- Para trazabilidad, usar siempre `tablas/tabla_02_paper_vs_mejor_local.csv` junto con la columna de objetivo.

## 6. Hallazgos preliminares de auditoria de datos

Resumen de `tablas/tabla_03_auditoria_datos.csv`:

- **Kepler**: dataset limpio en solapamientos entre splits (0/0/0 para train-val/train-test/val-test), pero con total local distinto al total del paper.
- **TESS**: diferencias de volumen respecto al paper y presencia de solapamientos entre splits (train-val=78, train-test=100, val-test=9), factor relevante para interpretar resultados.

Conclusión preliminar: parte de la brecha con el paper, sobre todo en TESS, no depende solo del optimizador/modelo, sino de diferencias de dataset y definicion de objetivo.

## 7. Figuras utiles para el paper

Se generaron y dejaron listas estas figuras:

1. `fig_01_paper_vs_local.png`: comparacion directa de metricas entre paper y mejor local.
2. `fig_02_evolucion_f1_mcc.png`: trayectoria experimental de F1/MCC por etapa.
3. `fig_03_top_configs_kepler.png`: ranking de configuraciones en Kepler.
4. `fig_04_top_configs_tess.png`: ranking de configuraciones en TESS.
5. `fig_05_auditoria_datos.png`: comparacion de conteos y solapamientos paper/local.

Adicionalmente se copiaron curvas PR de referencia (`figuras/ref_*`) para material de apoyo.

## 8. Como regenerar resultados de esta carpeta

Desde la raiz del proyecto:

```bash
/home/cristobal/miniconda3/envs/tf-gpu/bin/python entrega2/resultados_experimentales/generar_resultados_experimentales.py
```

Esto regenera tablas y figuras con los artefactos actuales de `resultados_*` y `mejores_resultados`.

## 9. Estado de la entrega

Esta entrega contiene resultados preliminares consolidados, con trazabilidad de configuraciones y entorno de ejecucion. El siguiente paso natural para el informe final es fijar un protocolo unico de comparacion estricta en TESS (mismo objetivo que paper) y reportar resultados finales sobre ese protocolo junto con analisis de sensibilidad por umbral.
