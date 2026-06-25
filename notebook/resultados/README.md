# Notebook de resultados XAI

Esta carpeta concentra el notebook principal y una copia ordenada de los resultados usados para discutir interpretabilidad.

## Archivos principales

- `xai_resultados_avanzados.ipynb`: notebook explicativo con tablas y figuras.
- `generar_notebook_resultados.py`: script para reconstruir `resultados/`.
- `resultados/tablas/`: tablas CSV copiadas desde `resultados_xai` y tablas nuevas de incertidumbre.
- `resultados/figuras/`: figuras XAI, matrices de focos y graficos de incertidumbre.
- `resultados/figuras_academicas/espanol/`: figuras con titulos y ejes academicos en espanol.
- `resultados/figuras_academicas/english/`: figuras equivalentes en ingles.
- `resultados/figuras_academicas/analisis.txt`: analisis general de desempeno, incertidumbre, fidelidad, focos de atencion y conclusiones.
- `resultados/figuras_academicas/conclusión_final.txt`: conclusion final lista para informe, con discusion de XAI y fidelidad.
- `resultados/figuras_academicas/figuras_academicas_explicacion_unaAuna.txt`: explicacion individual de cada figura principal y suplementaria.
- `resultados/figuras_academicas/fidelidad_ampliada/`: evaluacion ampliada tipo Faithfulness Correlation sobre 260 muestras estratificadas; incluye nota sobre Quantus y SHAP.

## Incertidumbre

`MaxLike` corresponde a la inferencia deterministica del modelo con `model.eval()` y dropout desactivado. Es la prediccion puntual de maxima verosimilitud usada como base.

`MC Dropout` ejecuta multiples inferencias con dropout activo. La media de esas predicciones resume la probabilidad predictiva y la desviacion estandar mide variabilidad entre muestras.

Para clasificacion binaria se usa:

- Incertidumbre predictiva: `H(E[p])`.
- Incertidumbre aleatorica: `E[H(p_t)]`.
- Incertidumbre epistemica: `H(E[p]) - E[H(p_t)]`.

La incertidumbre aleatorica representa ruido/inherente ambiguedad de los datos. La epistemica representa incertidumbre del modelo y deberia bajar con mas datos/modelos mejores.

## Focos de atencion

El conteo de focos usa Saliency target-aware en todo el test set. Se cuentan componentes conectados de alta relevancia en las vistas global y local. Es un analisis exploratorio para discutir si errores y aciertos presentan distinta cantidad o dispersion de zonas de atencion.

Las matrices de confusion de focos se reportan normalizadas por fila real. Es decir, cada fila suma 100% y el color representa la proporcion de ejemplos de una etiqueta real que cae en cada prediccion. El texto de cada celda mantiene el conteo `n`, la media de focos y la dispersion temporal de la atencion.

Tambien se generan violin plots academicos. El primer violin plot resume el `attention_risk_score`, que combina centralidad, dispersion, cantidad de focos, incertidumbre y confianza. El segundo muestra la distribucion de la cantidad de focos y la varianza de posicion de la atencion, anotando media y varianza por grupo.

## Metricas XAI adicionales

Ademas de las figuras originales, se generan metricas derivadas:

- `xai_faithfulness_auc_summary.csv`: AUC normalizada de la caida de confianza en la prueba de borrado.
- `xai_faithfulness_correlation_summary.csv`: Faithfulness Correlation estilo Bhatt et al. sobre perturbaciones aleatorias.
- `xai_method_agreement_summary.csv`: acuerdo top-10 entre metodos de explicabilidad.
- `xai_method_centrality_summary.csv`: fraccion de importancia central por metodo y vista.
- `xai_attention_quality_summary.csv`: score de calidad/riesgo de atencion por TP/TN/FP/FN.
- `xai_error_detection_metrics.csv`: ROC-AUC de metricas XAI/incertidumbre para separar errores de aciertos.

Estas metricas son utiles para comparar metodos y plantear trabajo futuro, por ejemplo usar estadisticos XAI como features de alerta de predicciones poco confiables.

## Faithfulness Correlation

Se agrego una implementacion local de Faithfulness Correlation, equivalente en objetivo a la metrica sugerida por Bhatt et al. (2021). Para cada ejemplo seleccionado se generan perturbaciones aleatorias que apagan 10% de los puntos de la vista global y local. Luego se calcula la correlacion entre:

- la suma de atribucion XAI en los puntos perturbados;
- la caida de confianza del modelo hacia la clase explicada.

Un valor positivo alto indica que los puntos considerados importantes por el metodo efectivamente producen mayor caida de confianza cuando se perturban. Un valor cercano a cero indica asociacion debil. Un valor negativo sugiere que el mapa puede estar resaltando regiones poco fieles para esa prediccion.

Resumen obtenido:

| dataset | method | valid_n | pearson_mean | spearman_mean | mean_confidence_drop |
| --- | --- | --- | --- | --- | --- |
| kepler | integrated_gradients | 5 | 0.2115 | 0.2462 | 0.0300 |
| kepler | occlusion | 5 | 0.2109 | 0.1594 | 0.0300 |
| kepler | consensus | 5 | 0.1961 | 0.1536 | 0.0300 |
| kepler | saliency | 5 | 0.1434 | 0.1699 | 0.0300 |
| kepler | gradcam_multiscale | 5 | 0.0496 | 0.0565 | 0.0300 |
| tess | occlusion | 6 | 0.4957 | 0.5138 | -0.0667 |
| tess | consensus | 6 | 0.3654 | 0.4209 | -0.0667 |
| tess | gradcam_multiscale | 6 | 0.3154 | 0.3736 | -0.0667 |
| tess | saliency | 6 | 0.2049 | 0.2050 | -0.0667 |
| tess | integrated_gradients | 6 | 0.1806 | 0.2574 | -0.0667 |

Limitaciones: esta metrica no prueba causalidad. Las perturbaciones con valor cero pueden sacar la curva de luz de la distribucion real, y en redes neuronales convolucionales no lineales existen interacciones entre puntos temporales que una correlacion por perturbacion no captura completamente. Por eso se reporta como evidencia cuantitativa complementaria, no como validacion absoluta del metodo XAI. Esto es consistente con la advertencia reciente de que las metricas de fidelidad suelen ser mas confiables en modelos lineales que en modelos no lineales profundos, como senalan Miró-Nicolau et al. (2025).

## Se podran usar estas explicaciones para mejorar el modelo?

Si, pero no conviene usarlas como etiquetas verdaderas. Lo mas seguro es tratarlas como senales auxiliares de diagnostico. Si un falso positivo mira fuera del transito, puede indicar ruido, mala normalizacion, evento secundario o una morfologia confundente. Si un falso negativo tiene atencion dispersa o poca importancia central, puede indicar centrado deficiente, baja senal-ruido o preprocesamiento inadecuado.

Como trabajo futuro, estas explicaciones podrian mejorar el desempeno mediante: auditoria de datos, seleccion de ejemplos dificiles para revision, ajuste de umbrales segun riesgo, entrenamiento de un detector auxiliar de predicciones poco confiables y regularizacion explicable que penalice atencion fuera de zonas fisicamente razonables. Esto siempre debe validarse con metricas en test, porque en una CNN no lineal los mapas XAI no prueban causalidad por si solos.
