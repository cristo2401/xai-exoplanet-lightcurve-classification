# Proyecto XAI - Entrega 2

Repositorio de trabajo para la clasificación automática de curvas de luz de exoplanetas utilizando datos de Kepler y TESS. El proyecto replica y adapta un pipeline basado en SE-CNN-RINet, incorpora auditoría de datos, búsqueda preliminar de configuraciones y generación reproducible de resultados experimentales.

Esta entrega incluye resultados preliminares, modelos guardados, métricas, figuras y scripts necesarios para reproducir las principales tablas utilizadas en el informe.

## Estructura del repositorio

```text
.
├── codigo/
│   ├── convertir_kepler_tfrecord_a_h5.py
│   ├── convertir_tess_tfrecord_a_h5.py
│   ├── kepler_official_split_and_hsearch.py
│   ├── tess_official_split_diagnostics.py
│   ├── tess_triage_hparam_search.py
│   └── export_mejores_resultados.py
│
├── tfrecords_KEPLER/
│   └── Datos Kepler en formato TFRecord
│
├── tfrecords_TESS/
│   └── Datos TESS en formato TFRecord
│
├── datos_procesados_h5/
│   ├── README.md
│   └── Archivos HDF5 generados localmente desde los TFRecords
│
├── mejores_resultados/
│   ├── modelos/
│   ├── metricas/
│   ├── mejores_vs_paper.csv
│   └── README.md
│
├── resultados_experimentales/
│   ├── tablas/
│   ├── figuras/
│   ├── generar_resultados_experimentales.py
│   └── README.md
│
├── resultados_xai/
│   ├── tablas/
│   ├── figuras/
│   └── README.md
│
├── notebook/
│   ├── xai_resultados_avanzados.ipynb
│   ├── generar_notebook_resultados.py
│   ├── README.md
│   └── resultados/
│       ├── tablas/
│       ├── figuras/
│       └── README.md
│
├── xai_mejores_modelos.py
├── xai_mejores_modelos.ipynb
├── requirements.txt
├── VALIDACION_REPRODUCIBILIDAD.txt
├── .gitignore
├── .gitattributes
└── README.md
```


## Nota Sobre El ZIP Liviano

Este paquete esta preparado para GitHub. No incluye archivos `.h5`, caches, ZIPs internos ni figuras intermedias/suplementarias duplicadas. Los datos crudos TFRecord, modelos, codigo, metricas y figuras principales se mantienen. Las salidas omitidas se regeneran con los comandos de este README. Ver tambien `CONTENIDO_ZIP_LIVIANO.txt` y `VALIDACION_REPRODUCIBILIDAD.txt`.

## Requisitos

- Python 3.10 o superior.
- Se recomienda Python 3.11 o 3.12.
- Entorno virtual o entorno Conda.
- Dependencias listadas en `requirements.txt`.
- GPU NVIDIA opcional para acelerar entrenamiento.
- El código puede ejecutarse en CPU, aunque con mayor tiempo de cómputo.

## Instalación

Desde la raíz del repositorio:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

En Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

También puede usarse Conda:

```bash
conda create -n xai-exoplanets python=3.12
conda activate xai-exoplanets
pip install -r requirements.txt
```

## Reproducción de resultados

Hay dos formas de ejecutar el proyecto: una rápida (artefactos principales) y una completa (incluye entrenamiento/búsqueda de configuraciones).

### Opción A: Reproducción rápida (recomendada para corregir/entregar)

Ejecuta esto en orden desde la raíz del repo:

```bash
# 0) Activar entorno e instalar dependencias (si aún no lo hiciste)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) Convertir TFRecord -> HDF5
python codigo/convertir_kepler_tfrecord_a_h5.py --output_h5 datos_procesados_h5/kepler_dataset.h5
python codigo/convertir_tess_tfrecord_a_h5.py --mode pc_only --output_h5 datos_procesados_h5/tess_dataset.h5
python codigo/convertir_tess_tfrecord_a_h5.py --mode triage --output_h5 datos_procesados_h5/tess_dataset_triage.h5

# 2) Entrenar/exportar modelos "champion" locales (Kepler + TESS)
python codigo/export_mejores_resultados.py

# 3) Generar tablas/figuras de resultados experimentales
python resultados_experimentales/generar_resultados_experimentales.py
```

Salida esperada:

```text
datos_procesados_h5/*.h5          # generado localmente; no se versiona en GitHub
mejores_resultados/modelos/*.pth
mejores_resultados/metricas/*.json
mejores_resultados/mejores_vs_paper.csv
resultados_experimentales/tablas/*.csv
resultados_experimentales/figuras/*.png
```

### Opción B: Reproducción completa (incluye entrenamiento + diagnósticos)

Usa esta ruta si quieres regenerar búsquedas y análisis intermedios:

```bash
# 1) Conversión de datos
python codigo/convertir_kepler_tfrecord_a_h5.py --output_h5 datos_procesados_h5/kepler_dataset.h5
python codigo/convertir_tess_tfrecord_a_h5.py --mode pc_only --output_h5 datos_procesados_h5/tess_dataset.h5
python codigo/convertir_tess_tfrecord_a_h5.py --mode triage --output_h5 datos_procesados_h5/tess_dataset_triage.h5

# 2) Kepler: entrenamiento y búsqueda de hiperparámetros
python codigo/kepler_official_split_and_hsearch.py

# 3) TESS: diagnóstico de split oficial + pruebas de fuga/definición de etiqueta
python codigo/tess_official_split_diagnostics.py

# 4) TESS triage: entrenamiento y búsqueda de hiperparámetros
python codigo/tess_triage_hparam_search.py

# 5) Exportar mejores modelos y métricas finales
python codigo/export_mejores_resultados.py

# 6) Generar tablas y figuras del informe
python resultados_experimentales/generar_resultados_experimentales.py
```

Salida esperada por etapa:

```text
resultados_kepler_hsearch/kepler_hsearch_summary.csv
resultados_tess_official_split/diagnostics_summary.csv
resultados_tess_triage_hsearch/hsearch_summary.csv
mejores_resultados/mejores_vs_paper.csv
resultados_experimentales/tablas/*.csv
resultados_experimentales/figuras/*.png
```

### Verificación rápida

```bash
ls resultados_kepler_hsearch
ls resultados_tess_official_split
ls resultados_tess_triage_hsearch
ls mejores_resultados/metricas
ls resultados_experimentales/tablas
```

## Análisis XAI

El repositorio incluye un flujo adicional de interpretabilidad para los mejores modelos guardados en `mejores_resultados/`. Este análisis no reentrena modelos: carga los checkpoints finales y genera explicaciones sobre ejemplos reales del conjunto de test.

Desde la raíz del repositorio:

```bash
python xai_mejores_modelos.py \
    --dataset both \
    --mc-samples 100 \
    --smooth-samples 32 \
    --ig-steps 64 \
    --max-examples 6 \
    --batch-size 512
```

También se puede ejecutar por dataset:

```bash
python xai_mejores_modelos.py --dataset kepler
python xai_mejores_modelos.py --dataset tess
```

El notebook `xai_mejores_modelos.ipynb` usa el mismo script y sirve para visualizar las tablas y figuras de forma interactiva.

Salida esperada:

```text
resultados_xai/tablas/xai_examples_summary.csv
resultados_xai/tablas/mc_dropout_summary.csv
resultados_xai/tablas/xai_importance_metrics.csv
resultados_xai/tablas/xai_faithfulness_metrics.csv
resultados_xai/tablas/xai_calibration_summary.csv
resultados_xai/tablas/xai_reliability_bins.csv
resultados_xai/tablas/xai_counterfactual_summary.csv
resultados_xai/tablas/xai_outcome_summary.csv
resultados_xai/tablas/xai_attention_focus_by_sample.csv
resultados_xai/tablas/xai_attention_focus_confusion_matrix.csv
resultados_xai/tablas/xai_attention_focus_outcome_summary.csv
resultados_xai/figuras/*_xai_maps.png
resultados_xai/figuras/*_mc_dropout_uncertainty.png
resultados_xai/figuras/*_mc_dropout_scatter.png
resultados_xai/figuras/*_aggregate_importance_selected_examples.png
resultados_xai/figuras/*_faithfulness_deletion_curves.png
resultados_xai/figuras/*_reliability_calibration.png
resultados_xai/figuras/*_counterfactual_effects.png
resultados_xai/figuras/*_outcome_uncertainty_summary.png
resultados_xai/figuras/*_attention_focus_confusion_matrix.png
resultados_xai/figuras/*_attention_focus_outcome_boxplot.png
```

Métodos incluidos:

- `Saliency Maps`: importancia por gradiente respecto a cada punto temporal y clase explicada.
- `SmoothGrad`: promedio de gradientes sobre entradas con ruido pequeño para reducir mapas espurios.
- `Integrated Gradients`: atribución acumulada desde una línea base neutra hasta la curva real.
- `Grad-CAM 1D`: mapa de relevancia temporal usando activaciones convolucionales. Se interpreta como diagnóstico secundario, porque puede ser grueso por el pooling temporal.
- `Grad-CAM multiescala`: combina mapas desde varias capas convolucionales para recuperar evidencia temporal más fina.
- `Occlusion Sensitivity`: importancia por cambio de probabilidad al ocultar ventanas temporales.
- `Mapa de consenso`: combinación de SmoothGrad, Integrated Gradients, Occlusion y Grad-CAM multiescala para obtener una explicación más estable.
- `Fidelidad por borrado`: prueba cuantitativa que mide cuánto cae la confianza al eliminar las zonas más relevantes.
- `Faithfulness Correlation`: métrica formal de fidelidad estilo Bhatt et al. (2021), calculada mediante perturbaciones aleatorias y correlación entre atribución XAI y cambio de confianza.
- `Contrafactuales simples`: intervenciones sobre la ventana central del tránsito para medir cambios de probabilidad.
- `Conteo de focos de atención`: conteo exploratorio de componentes conectados de alta relevancia en Saliency para todo el test set.
- `Calibración post-hoc`: temperature scaling usando validación; no modifica los pesos del modelo.
- `MC Dropout`: estimación de incertidumbre por múltiples inferencias con dropout activo.
- `xai_importance_metrics.csv`: métricas para cuantificar concentración de relevancia cerca del tránsito y solapamiento entre métodos.
- `xai_faithfulness_metrics.csv`: métricas de caída de confianza al borrar regiones importantes.
- `xai_calibration_summary.csv`: ECE, MCE, Brier y NLL antes/después de calibrar.
- `xai_counterfactual_summary.csv`: cambios de probabilidad bajo contrafactuales.
- `xai_outcome_summary.csv`: incertidumbre y confianza por TP, TN, FP y FN.
- `xai_attention_focus_by_sample.csv`: conteo de focos y dispersión temporal por muestra.
- `xai_attention_focus_confusion_matrix.csv`: resumen de focos por celda de matriz de confusión, con proporciones normalizadas por etiqueta real.
- `xai_attention_focus_outcome_summary.csv`: resumen por TP, TN, FP y FN para discutir como trabajo futuro.

### Notebook ordenado de resultados

Para revisar los resultados XAI en una vista mas limpia, se incluye:

```text
notebook/xai_resultados_avanzados.ipynb
```

Este notebook usa `notebook/resultados/`, que contiene copias ordenadas de tablas y figuras, mas resultados adicionales para MaxLike, MC Dropout, incertidumbre aleatorica, incertidumbre epistemica y focos de atencion.

Para reconstruir esa carpeta:

```bash
python notebook/generar_notebook_resultados.py
```

Métricas adicionales incluidas en `notebook/resultados/tablas/`:

- `xai_faithfulness_auc_summary.csv`: AUC normalizada de fidelidad por borrado.
- `xai_faithfulness_correlation_summary.csv`: Faithfulness Correlation por método XAI.
- `xai_method_agreement_summary.csv`: acuerdo top-10 entre métodos XAI.
- `xai_method_centrality_summary.csv`: centralidad de importancia por método y vista.
- `xai_attention_quality_summary.csv`: score de calidad/riesgo de atención por outcome.
- `xai_error_detection_metrics.csv`: ROC-AUC de métricas XAI/incertidumbre para separar errores de aciertos.

Las explicaciones XAI se proponen como trabajo futuro para mejorar el desempeño del modelo de forma indirecta: auditoría de datos, detección de predicciones poco confiables, ajuste de umbrales, revisión de ejemplos difíciles y posibles mejoras de preprocesamiento. No se usan como etiquetas verdaderas, porque en redes neuronales no lineales los mapas de importancia no prueban causalidad por sí solos.

Figuras académicas bilingües:

```text
notebook/resultados/figuras_academicas/
├── indice_figuras_academicas.csv
├── analisis.txt
├── conclusión_final.txt
├── figuras_academicas_explicacion_unaAuna.txt
├── fidelidad_ampliada/
├── espanol/
│   ├── figuras_principales/
│   └── material_suplementario/
└── english/
    ├── main_figures/
    └── supplementary_figures/
```

Estas carpetas contienen las figuras principales listas para paper y todo el material suplementario con nombres académicos en español e inglés. El archivo `indice_figuras_academicas.csv` permite rastrear cada figura desde su nombre original hasta su versión académica. El archivo `analisis.txt` resume las conclusiones generales del bloque XAI, `conclusión_final.txt` contiene la discusión final lista para informe, `figuras_academicas_explicacion_unaAuna.txt` explica cada figura principal y suplementaria, y `fidelidad_ampliada/` contiene la evaluación reforzada de Faithfulness Correlation sobre 260 muestras.



## Scripts principales

| Script | Descripción |
|---|---|
| `codigo/convertir_kepler_tfrecord_a_h5.py` | Convierte los datos de Kepler desde TFRecord a HDF5. |
| `codigo/convertir_tess_tfrecord_a_h5.py` | Convierte los datos de TESS desde TFRecord a HDF5. Permite modo `pc_only` o `triage`. |
| `codigo/kepler_official_split_and_hsearch.py` | Ejecuta búsqueda preliminar de configuraciones en Kepler. |
| `codigo/tess_official_split_diagnostics.py` | Diagnostica particiones, etiquetas y posibles problemas de TESS. |
| `codigo/tess_triage_hparam_search.py` | Ejecuta búsqueda de hiperparámetros para TESS en modo triage. |
| `codigo/export_mejores_resultados.py` | Exporta modelos, métricas y comparación contra el paper de referencia. |
| `resultados_experimentales/generar_resultados_experimentales.py` | Genera tablas y figuras para la entrega. |
| `xai_mejores_modelos.py` | Genera análisis XAI con Saliency, SmoothGrad, Integrated Gradients, Grad-CAM 1D/multiescala, Occlusion, Consenso, calibración, contrafactuales y MC Dropout. |

## Resultados preliminares

Los principales resultados se encuentran en:

```text
mejores_resultados/
resultados_experimentales/
```

Archivos relevantes:

| Archivo | Descripción |
|---|---|
| `mejores_resultados/README.md` | Resumen de los mejores modelos obtenidos. |
| `mejores_resultados/mejores_vs_paper.csv` | Comparación entre resultados locales y el paper de referencia. |
| `resultados_experimentales/README.md` | Resumen de resultados experimentales preliminares. |
| `resultados_experimentales/tablas/` | Tablas en formato CSV. |
| `resultados_experimentales/figuras/` | Figuras utilizadas en el informe. |

## Notas metodológicas

- Kepler se evalúa principalmente como clasificación binaria de candidato planetario vs no candidato.
- TESS se analiza bajo dos definiciones:
  - `pc_only`: `PC` vs resto.
  - `triage`: `PC+EB` vs resto.
- Los resultados de TESS bajo `triage` no deben compararse directamente con resultados estrictos `PC` vs resto.
- La evaluación considera métricas robustas a desbalance, como F1, MCC y AP.
- Se realiza auditoría de datos para revisar distribución de clases, duplicados y posibles solapamientos entre particiones.

## Consideraciones sobre datos

Los TFRecords necesarios para esta entrega se incluyen en `tfrecords_KEPLER/` y `tfrecords_TESS/`.

Los archivos HDF5 de `datos_procesados_h5/` no se incluyen en GitHub porque son artefactos generados y pueden superar limites practicos de tamano de archivo. Se regeneran con:

```bash
python codigo/convertir_kepler_tfrecord_a_h5.py --output_h5 datos_procesados_h5/kepler_dataset.h5
python codigo/convertir_tess_tfrecord_a_h5.py --mode pc_only --output_h5 datos_procesados_h5/tess_dataset.h5
python codigo/convertir_tess_tfrecord_a_h5.py --mode triage --output_h5 datos_procesados_h5/tess_dataset_triage.h5
```

Si en una version futura se desean subir archivos mayores a 100 MB, se debe usar Git LFS. En esta entrega el ZIP fue limpiado para no incluir archivos individuales sobre ese limite.

## Reproducibilidad

Todas las rutas utilizadas por los scripts están definidas de forma relativa a la raíz del repositorio. Esto permite ejecutar el proyecto en distintos equipos sin depender de rutas locales absolutas.

El repositorio incluye los scripts necesarios para reconstruir los principales artefactos de la entrega: modelos, métricas, tablas y figuras.

## Repositorio

El repositorio debe estar visible para la profesora y el ayudante de la asignatura. Si se utiliza un repositorio privado, se deben otorgar permisos de lectura a las cuentas correspondientes.
