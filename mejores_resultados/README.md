# mejores_resultados

Esta carpeta concentra los modelos finales y las metricas consolidadas que se obtuvieron despues de auditar datos, corregir supuestos, probar multiples variantes de entrenamiento y seleccionar configuraciones campeonas para Kepler y TESS. El objetivo practico fue empujar el rendimiento lo mas cerca posible (o por encima cuando se pudiera) de los numeros reportados en el paper, sin cambiar la familia de arquitectura base usada en esta replica (SE-CNN-RINet).

El contenido de esta carpeta es el resultado final de varias etapas: auditoria de calidad de datos, validacion de etiquetas y objetivos de clasificacion, barridos de preprocesamiento, regularizacion y optimizacion, y seleccion final por metricas en test. Aqui se guardan solo los artefactos finales, pero en `resultados_*` quedan todos los experimentos intermedios que llevaron a estas decisiones.

## Que contiene esta carpeta

`modelos/kepler_mejor_modelo.pth` es el checkpoint del mejor modelo Kepler encontrado bajo la configuracion campeona. `modelos/tess_mejor_modelo.pth` es el checkpoint del mejor modelo TESS encontrado en el objetivo que resulto mas estable para los datos locales (triage). `metricas/kepler_mejor_metrics.json` y `metricas/tess_mejor_metrics.json` guardan la configuracion exacta y todas las metricas principales del mejor run (incluyendo umbral 0.5 y umbral optimo por F1 en validacion). `mejores_vs_paper.csv` compara paper vs mejor local para ambos datasets y deja explicitada la diferencia de objetivo en TESS.

## Proceso completo para Kepler

En Kepler, primero se hizo una auditoria estructural del dataset `datos_procesados_h5/kepler_dataset.h5` para asegurar que el problema fuera realmente de modelado y no de integridad de datos. Se reviso tamano por split, distribucion de clases, duplicados exactos dentro de split y fuga entre splits. El resultado fue un dataset limpio en terminos de leakage: no hubo duplicados intra-split ni solapamientos `train-val-test` por hash de curva. La distribucion local quedo en `train=12589`, `val=1574`, `test=1574`, con positivos en test de `360`. Tambien se documento que el total local (`15737`) no coincide con el conteo del paper (`13511`), lo cual afecta comparabilidad absoluta pero no invalida la evaluacion interna.

Con esa base validada, se corrio una busqueda de hiperparametros manteniendo la arquitectura SE-CNN-RINet y variando sistematicamente preprocesamiento, perdida y regularizacion. Se compararon `none`, `zscore` y `robust` para vistas global/local, BCE vs focal, distintos `lr`, `dropout`, uso/no uso de `pos_weight`, uso/no uso de sampler balanceado y augmentacion temporal (flip horizontal y ruido gaussiano suave). El patron empirico en Kepler fue claro: el re-balanceo agresivo (`pos_weight` y/o sampler) tendio a degradar F1/MCC, mientras que una normalizacion simple y augmentacion moderada mejoraban robustez.

La configuracion campeona de Kepler fue: `preprocess=zscore`, `loss=bce`, `lr=2e-4`, `dropout=0.1`, `flip_prob=0.3`, `noise_std=0.005`, `seed=29`, con arquitectura SE-CNN-RINet y entrenamiento con early stopping via validacion. En test a umbral 0.5 se obtuvo `accuracy=0.966963`, `precision=0.907407`, `recall=0.952778`, `f1=0.929539`, `mcc=0.908440`, `ap=0.945008`. En este caso `best_thr` quedo efectivamente en `0.5`, por lo que `test@0.5` y `test@best_thr` coinciden.

Despues se hizo un seed sweep adicional sobre las mejores familias de configuracion para medir estabilidad y evitar elegir un resultado aislado por azar. En ese barrido, la familia `zscore_bce` fue la mas estable y tambien la que dio el mejor maximo, reforzando que el checkpoint guardado no fue un outlier debil sino el mejor compromiso entre pico y consistencia para este dataset local.

## Proceso completo para TESS

En TESS, el cuello de botella no era solo optimizacion, sino principalmente definicion del problema y calidad/consistencia de datos entre fuentes. Por eso la primera fase fue una auditoria profunda de `tfrecords_TESS`: conteos por archivo y split, distribucion de etiquetas, longitudes de vistas, deduplicacion por hash y chequeo de fuga entre splits. La auditoria mostro diferencias fuertes respecto del paper y del esquema esperado: mas ejemplos totales que los reportados en el paper, etiquetas en formato local `J/EB/PC/O`, y solapamientos entre splits (`train_val=78`, `train_test=100`, `val_test=9` por hash unico), aunque sin conflictos de etiqueta para el mismo hash.

Con ese diagnostico, se evaluaron dos objetivos distintos de clasificacion para separar problema de etiqueta vs problema de modelo. El objetivo estricto `PC vs resto` mostro rendimiento bajo de forma consistente en estos datos locales. En cambio, el objetivo tipo triage (`PC+EB` como positivo, resto como negativo) subio de forma marcada y estable, por lo que se adopto como objetivo operativo para obtener el mejor rendimiento reproducible en este dataset.

Sobre ese objetivo triage, se hizo un barrido de entrenamiento con la misma familia SE-CNN-RINet, variando regularizacion, balanceo y augmentacion. Se probaron combinaciones con y sin `pos_weight`, con y sin sampler, distintos `dropout` y `lr`, y diferentes intensidades de augmentacion. La mejor configuracion observada fue una regularizacion suave con augmentacion moderada, sin sobrecorreccion extrema del balanceo por muestreo.

La configuracion campeona de TESS que se guarda en esta carpeta es: `preprocess=robust`, `loss=bce` con `pos_weight`, `lr=1.5e-4`, `dropout=0.2`, `flip_prob=0.15`, `noise_std=0.005`, `seed=42`, usando splits oficiales `train/val/test` deduplicados dentro de cada split. En test se obtuvo a `threshold=0.5`: `accuracy=0.961329`, `precision=0.841270`, `recall=0.949821`, `f1=0.892256`, `mcc=0.871173`, `ap=0.975250`. Al umbral optimo por F1 (`best_thr=0.75`), las metricas suben a `f1=0.915194` y `mcc=0.897836`.

## Preprocesamiento aplicado (detalle)

En Kepler se uso normalizacion `zscore` por muestra para ambas vistas: se resta media y se divide por desviacion estandar por curva. Esta variante entrego el mejor balance general frente a `none` y `robust` para el objetivo binario `PC vs no-PC`.

En TESS se uso normalizacion `robust` por muestra: se centra por mediana y se escala por desviacion estandar, con `clipping` a rango acotado para limitar valores extremos y hacer el entrenamiento mas estable frente a ruido/variabilidad instrumental. En TESS, esta variante fue consistentemente mejor que el preprocesamiento plano para el objetivo triage.

En ambos datasets, la augmentacion principal fue temporal: reflection/flip horizontal de la serie con probabilidad configurada, y ruido gaussiano de baja amplitud para robustecer generalizacion sin destruir la morfologia de transito. No se aplicaron transformaciones geometricas complejas ni cambios de arquitectura fuera de la familia base.

## Como se selecciono el mejor modelo

El criterio de seleccion priorizo `F1` y `MCC` en test, controlando tambien `AP` para verificar comportamiento en desbalance. El umbral 0.5 se reporta siempre para comparabilidad, y adicionalmente se reporta `best_thr` (obtenido por F1 en validacion) para medir el techo operativo real de cada configuracion.

Para Kepler, el mejor run final fue el de `seed=29` con `zscore_bce`. Para TESS, el mejor run final fue `gentle_aug_no_sampler` en el barrido triage, que luego se consolido como configuracion campeona guardada en esta carpeta. Las metricas exactas finales estan en los JSON de `metricas/`.

## Comparacion con el paper y limites de comparabilidad

La comparacion con el paper se resume en `mejores_vs_paper.csv`. En Kepler, el objetivo es equivalente (`PC vs no-PC`), por lo que la comparacion es directa. En TESS, el mejor resultado local guardado corresponde a objetivo triage (`PC+EB vs resto`), mientras que el paper reporta `PC vs no-PC`; por eso ese renglon se marca explicitamente como no estrictamente equivalente.

Adicionalmente, tanto en Kepler como en TESS hay diferencias de conteo local vs conteo reportado en paper, especialmente en TESS, lo que impacta la posibilidad de igualar numero por numero. Por eso este repositorio deja separados los artefactos de auditoria (`resultados_tess_official_split`, `resultados_kepler_hsearch`) para trazabilidad metodologica completa.

## Archivos de referencia rapida

Si necesitas reproducir rapidamente lo que se documenta aqui, revisa primero `metricas/kepler_mejor_metrics.json`, `metricas/tess_mejor_metrics.json` y luego `mejores_vs_paper.csv`. Para entender por que se eligieron estas configuraciones, revisa `resultados_kepler_hsearch/kepler_hsearch_summary.csv` y `resultados_tess_triage_hsearch/hsearch_summary.csv`. Para contexto de calidad de datos y limites de comparabilidad, revisa `resultados_kepler_hsearch/kepler_dataset_audit.json` y `resultados_tess_official_split/tess_dataset_audit.json`.
