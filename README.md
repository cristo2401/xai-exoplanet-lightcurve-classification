# Clasificación y explicación de curvas de luz Kepler/TESS

Proyecto académico de deep learning para clasificar curvas de luz y estudiar **qué
señales usa el modelo y cuándo sus predicciones son poco confiables**. Replica y
adapta un pipeline SE-CNN-RINet, con auditoría de datos, búsqueda de configuraciones
y análisis de interpretabilidad e incertidumbre.

Se incluyen código, modelos guardados, métricas y figuras de experimentos previos.
La revisión de estas salidas no requiere entrenar; reconstruirlas sí requiere
dependencias, conversión de datos y cómputo.

## Recorrido recomendado

| Si quieres… | Empieza por… |
|---|---|
| Entender los resultados de clasificación | [Resumen de modelos](mejores_resultados/README.md) y [métricas CSV](mejores_resultados/mejores_vs_paper.csv) |
| Revisar los supuestos y el preprocesamiento | [Código](codigo/) y [notas metodológicas](#alcance-y-limitaciones) |
| Explorar explicaciones e incertidumbre | [Notebook de resultados](notebook/xai_resultados_avanzados.ipynb) y [tablas XAI](notebook/resultados/tablas/) |
| Reproducir los experimentos | [Guía detallada](REPRODUCIBILITY.md) y [registro previo de validación](VALIDACION_REPRODUCIBILIDAD.txt) |

## Qué contiene

- Conversión TFRecord → HDF5 y diagnósticos de etiquetas, duplicados y particiones.
- Modelos de clasificación y búsqueda preliminar de hiperparámetros.
- Saliency, SmoothGrad, Integrated Gradients, Grad-CAM y Occlusion.
- MC Dropout, calibración de probabilidades y evaluación de fidelidad de atribuciones.
- Tablas y figuras para analizar errores y comparar explicaciones.

## Resultados guardados

Valores de las filas `mejor_local` en
[`mejores_vs_paper.csv`](mejores_resultados/mejores_vs_paper.csv), con umbral 0,5:

| Dataset | Objetivo | F1 | MCC | AP |
|---|---|---:|---:|---:|
| Kepler | Candidato planetario (PC) frente al resto | 0,9295 | 0,9084 | 0,9450 |
| TESS | PC + binarias eclipsantes (EB) frente al resto | 0,8923 | 0,8712 | 0,9752 |

Son resultados locales guardados. **Las dos tareas tienen etiquetas distintas**:
TESS usa un objetivo de triage, por lo que sus cifras no se comparan directamente
con una tarea estricta de planetas frente al resto ni con los resultados del paper
para esa tarea. No se afirma superioridad sobre el trabajo de referencia.

## Instalación y reproducción

El entorno incluye PyTorch, TensorFlow, NumPy, pandas, scikit-learn, Matplotlib y
h5py. La guía propone Python 3.11 o 3.12 y ejecución desde la raíz del repositorio.

```bash
git clone https://github.com/cristo2401/xai-exoplanet-lightcurve-classification.git
cd xai-exoplanet-lightcurve-classification
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

La descarga incluye TFRecords y modelos. Los archivos HDF5 se generan localmente.
La [guía de reproducción](REPRODUCIBILITY.md) distingue conversión, entrenamiento,
exportación de métricas y análisis XAI. Su ruta denominada «artefactos principales»
también entrena/exporta modelos: no es un smoke test de pocos segundos.

## Alcance y limitaciones

- Los resultados son de un proyecto académico y dependen del split, las etiquetas,
  el preprocesamiento y la selección de configuraciones.
- Los mapas de importancia ayudan a auditar modelos; no prueban causalidad física.
- Una explicación visual plausible no sustituye las pruebas de fidelidad ni la
  evaluación sobre muestras independientes.
- Las dependencias usan versiones mínimas, no un entorno completamente bloqueado.
  El registro previo de reproducibilidad no garantiza resultados idénticos en
  cualquier versión o hardware.
- La documentación de origen/licencia de los datos y de la referencia SE-CNN-RINet
  debe completarse antes de reutilizar o redistribuir materiales de terceros.

Los comandos, métodos y detalles del trabajo se conservan en
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).
