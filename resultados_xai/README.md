# Resultados XAI

Esta carpeta contiene explicaciones visuales para los mejores modelos guardados en `mejores_resultados/`.

## Metodos usados

**Saliency Maps**: calcula el gradiente absoluto del logit de la clase explicada respecto a cada punto de la curva de luz. Los puntos mas intensos indican zonas donde pequenos cambios de entrada afectan mas la prediccion.

**SmoothGrad**: promedia varios Saliency Maps calculados sobre versiones levemente perturbadas de la misma curva. Reduce ruido de gradiente y entrega mapas mas estables visualmente.

**Integrated Gradients**: acumula gradientes desde una linea base neutra hasta la curva real. Es menos ruidoso que el gradiente simple y ayuda a confirmar si la relevancia se sostiene al construir la senal completa.

**Grad-CAM 1D**: usa activaciones y gradientes de la ultima convolucion 1D de cada rama del modelo. En esta arquitectura se interpreta como diagnostico secundario, porque el pooling temporal puede generar mapas gruesos o desplazados, especialmente en TESS.

**Grad-CAM multiescala**: combina mapas generados desde varias capas convolucionales. La idea es recuperar evidencia temporal mas fina desde capas tempranas y compararla con el Grad-CAM clasico.

**Occlusion Sensitivity**: enmascara ventanas temporales y mide cuanto cambia la probabilidad del modelo. Es una prueba directa de sensibilidad: si tapar una zona cambia mucho la prediccion, esa zona es importante.

**Mapa de consenso**: combina SmoothGrad, Integrated Gradients, Occlusion y Grad-CAM multiescala. Se usa como resumen visual estable, mientras que la fidelidad final se valida con la prueba de borrado. En los resultados actuales, Occlusion es el metodo mas fuerte por intervencion.

**Fidelidad por borrado**: borra progresivamente los puntos mas importantes segun cada mapa y mide la caida de confianza hacia la clase explicada. Un metodo mas fiel deberia producir una caida mas rapida.

**Contrafactuales simples**: modifican la ventana central del transito, por ejemplo aplanandola, profundizandola o reduciendola, y miden como cambia la probabilidad. Esto ayuda a discutir si la decision depende realmente de la morfologia del transito.

**Conteo de focos de atencion**: calcula Saliency target-aware para todo el test set y cuenta componentes conectados de alta relevancia en las vistas global y local. La idea es medir si los aciertos tienen pocos focos compactos y si los errores presentan atencion mas dispersa.

**Calibracion de probabilidad**: ajusta una temperatura escalar usando el split de validacion. No cambia los pesos del modelo; solo permite reportar probabilidades calibradas, ECE, Brier, NLL y diagramas de confiabilidad.

**MC Dropout**: ejecuta el modelo 100 veces con dropout activo durante inferencia. La media resume la probabilidad estimada y la desviacion estandar aproxima la incertidumbre predictiva.

## Como interpretar las figuras

Cada figura muestra la vista global y local de una curva de luz. El color de fondo es el mapa de importancia: regiones mas brillantes tienen mayor relevancia para la clase explicada. En ejemplos positivos se explica evidencia a favor de la clase positiva; en ejemplos negativos se explica evidencia a favor de la clase negativa. Esto evita comparar todos los casos contra el mismo logit positivo.

La banda verde marca una ventana central alrededor del transito fase-plegado. No es una regla absoluta, pero sirve como referencia visual para discutir si la relevancia del modelo cae cerca de la region esperada.

## Lectura cuantitativa

Ademas de las figuras individuales, se genera `xai_importance_metrics.csv`. Esta tabla mide, para cada ejemplo y cada vista, el indice del pico de relevancia, la distancia del pico al centro, la fraccion de importancia dentro de la ventana central y el solapamiento entre los puntos mas relevantes de distintos metodos. Tambien se genera `xai_faithfulness_metrics.csv`, que mide cuanto cae la confianza al borrar las regiones mas relevantes.

Las tablas `xai_calibration_summary.csv` y `xai_reliability_bins.csv` cuantifican calibracion antes y despues de temperature scaling. `xai_counterfactual_summary.csv` mide cambios de probabilidad bajo intervenciones simples. `xai_outcome_summary.csv` resume incertidumbre y confianza por TP, TN, FP y FN. Las tablas `xai_attention_focus_*.csv` resumen cuantas zonas de atencion aparecen por muestra y como varia esa cantidad dentro de cada celda de la matriz de confusion.

## Se podran usar estas explicaciones para mejorar el modelo?

Si, pero no de forma directa como si los mapas XAI fueran etiquetas verdaderas. La forma mas segura es usarlas como senales auxiliares de diagnostico. Por ejemplo, si un falso positivo concentra mucha atencion fuera de la ventana central del transito, eso puede indicar ruido, mala normalizacion, un evento secundario o una morfologia que el modelo esta confundiendo. Si un falso negativo muestra atencion muy dispersa o baja centralidad, puede sugerir que el transito quedo mal centrado, que la senal esta degradada o que el preprocesamiento elimino informacion util.

En una etapa futura, estas explicaciones podrian mejorar el desempeno de cuatro maneras. Primero, como auditoria de datos: revisar manualmente muestras con alta incertidumbre, muchos focos dispersos o baja fidelidad para detectar etiquetas dudosas, duplicados o curvas mal preprocesadas. Segundo, como detector de riesgo: entrenar un modelo auxiliar que use `total_focus_count`, entropia de atencion, centralidad, Faithfulness Correlation, MC Dropout y confianza para decidir cuando una prediccion debe aceptarse, revisarse o cambiar de umbral. Tercero, como guia de preprocesamiento: mejorar centrado, suavizado, resampling o ventanas locales cuando los mapas muestran que el modelo mira sistematicamente regiones no fisicas. Cuarto, como regularizacion explicable: penalizar atencion fuera de la zona esperada del transito, siempre que esa restriccion este justificada fisicamente y se valide en un test independiente.

La limitacion principal es que XAI no prueba causalidad por si solo. En redes neuronales no lineales, un mapa puede verse razonable y aun asi no ser completamente fiel al comportamiento interno del modelo. Por eso, cualquier mejora basada en XAI debe validarse con metricas de clasificacion en test, calibracion e idealmente con pruebas de fidelidad como borrado, Occlusion y Faithfulness Correlation. En esta entrega se deja planteado como trabajo futuro: usar las explicaciones no para reemplazar el clasificador, sino para mejorar el control de calidad, seleccionar ejemplos dificiles y construir alertas de predicciones poco confiables.

## Archivos

- `figuras/*_xai_maps.png`: curvas con Saliency, SmoothGrad, Integrated Gradients, Occlusion, Grad-CAM 1D, Grad-CAM multiescala y Consenso.
- `figuras/*_mc_dropout_uncertainty.png`: resumen visual de incertidumbre.
- `figuras/*_mc_dropout_scatter.png`: probabilidad vs incertidumbre para todo el test.
- `figuras/*_aggregate_importance_selected_examples.png`: importancia promedio de los ejemplos seleccionados.
- `figuras/*_faithfulness_deletion_curves.png`: caida de confianza al borrar puntos importantes.
- `figuras/*_reliability_calibration.png`: diagrama de confiabilidad antes y despues de calibracion.
- `figuras/*_counterfactual_effects.png`: efecto promedio de intervenciones contrafactuales.
- `figuras/*_outcome_uncertainty_summary.png`: incertidumbre media por tipo de resultado.
- `figuras/*_attention_focus_confusion_matrix.png`: matriz de confusion normalizada por fila real y anotada con conteos, media y desviacion de focos.
- `figuras/*_attention_focus_outcome_boxplot.png`: distribucion de focos y dispersion temporal por TP/TN/FP/FN.
- `tablas/xai_examples_summary.csv`: ejemplos explicados y sus metricas principales.
- `tablas/mc_dropout_summary.csv`: estadisticas MC Dropout para todo el test set procesado.
- `tablas/xai_importance_metrics.csv`: metricas cuantitativas de concentracion y solapamiento de relevancia.
- `tablas/xai_faithfulness_metrics.csv`: prueba cuantitativa de fidelidad por borrado.
- `tablas/xai_calibration_summary.csv`: ECE, MCE, Brier y NLL antes/despues de calibrar.
- `tablas/xai_reliability_bins.csv`: bins usados para los diagramas de confiabilidad.
- `tablas/xai_counterfactual_summary.csv`: cambios de probabilidad por contrafactual.
- `tablas/xai_outcome_summary.csv`: resumen por TP, TN, FP y FN.
- `tablas/xai_attention_focus_by_sample.csv`: conteo de focos por muestra.
- `tablas/xai_attention_focus_confusion_matrix.csv`: resumen tipo matriz de confusion, incluyendo proporcion normalizada por etiqueta real.
- `tablas/xai_attention_focus_outcome_summary.csv`: resumen agregado por TP/TN/FP/FN.
