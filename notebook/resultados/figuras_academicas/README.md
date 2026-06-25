# Figuras academicas XAI

Esta carpeta contiene dos versiones de las figuras listas para informe o paper:

- `espanol/`: nombres, titulos y organizacion en espanol.
- `english/`: nombres, titulos y organizacion en ingles.
- `imagenes_que_faltan_en_ingles/`: figuras en ingles con los mismos nombres de archivo usados originalmente, pensadas para reemplazo directo en Overleaf sin cambiar rutas.

## Estructura

```text
figuras_academicas/
├── indice_figuras_academicas.csv
├── analisis.txt
├── conclusión_final.txt
├── figuras_academicas_explicacion_unaAuna.txt
├── fidelidad_ampliada/
├── imagenes_que_faltan_en_ingles/
├── espanol/
│   ├── figuras_principales/
│   └── material_suplementario/
└── english/
    ├── main_figures/
    └── supplementary_figures/
```

## Figuras principales

Las carpetas `figuras_principales/` y `main_figures/` contienen las figuras mas recomendables para el cuerpo del informe/paper:

- matriz de confusion normalizada por fila real con resumen de focos XAI;
- violin plot del score de riesgo de atencion XAI por resultado;
- violin plot de cantidad de focos y varianza/dispersion temporal de atencion por resultado.

Estas figuras fueron regeneradas con mayor resolucion y con rotulos mas formales.

## Material suplementario

Las carpetas `material_suplementario/` y `supplementary_figures/` contienen todas las figuras generadas durante el analisis XAI, copiadas con nombres academicos en espanol e ingles. Esto incluye mapas XAI individuales, curvas de fidelidad, calibracion, incertidumbre, contrafactuales, centralidad, error detection y comparaciones MaxLike/MC Dropout.

`indice_figuras_academicas.csv` permite rastrear cada figura desde su archivo original hasta su version academica en ambos idiomas.

## Analisis general

El archivo `analisis.txt` entrega una lectura global de los resultados XAI. Resume desempeno predictivo, matrices normalizadas, comportamiento de los focos de atencion, incertidumbre MaxLike/MC Dropout, fidelidad, contrafactuales, calibracion, limitaciones y conclusiones generales.

## Conclusion final

El archivo `conclusión_final.txt` contiene una seccion lista para informe. Responde explicitamente que se logro explicar, si las explicaciones ayudan a interpretar el modelo, por que ocurren, como podrian mejorar el sistema de ML y como se interpreta la fidelidad de las explicaciones.

## Fidelidad ampliada

La carpeta `fidelidad_ampliada/` contiene una evaluacion tipo Faithfulness Correlation sobre 260 muestras estratificadas del test set. Esta carpeta documenta que Quantus no fue usado literalmente, explica por que SHAP se elimina del paper, y entrega tablas/figuras nuevas para defender mejor la fidelidad XAI.

## Imagenes faltantes en ingles

La carpeta `imagenes_que_faltan_en_ingles/` contiene siete figuras listas para Overleaf con los mismos nombres pedidos originalmente. Las tres figuras de fidelidad ampliada fueron regeneradas con rotulos en ingles; las cuatro figuras XAI restantes corresponden a las versiones academicas en ingles ya generadas en el material suplementario.

## Explicacion una a una

El archivo `figuras_academicas_explicacion_unaAuna.txt` explica cada figura principal y suplementaria, indicando que muestra, como interpretarla y cual es la conclusion especifica para Kepler o TESS.
