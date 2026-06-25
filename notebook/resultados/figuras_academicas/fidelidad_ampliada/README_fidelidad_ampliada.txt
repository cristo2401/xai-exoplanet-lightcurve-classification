# Fidelidad XAI ampliada y nota sobre Quantus/SHAP

Esta carpeta documenta una mejora específica al análisis de fidelidad. En la versión inicial, la fidelidad se calculó sobre los ejemplos seleccionados para visualización, aproximadamente 6 por dataset. Eso era suficiente como evidencia preliminar, pero era débil como respaldo cuantitativo porque la muestra era pequeña.

Para mejorar esto, se agregó una evaluación ampliada tipo **Faithfulness Correlation** sobre una muestra estratificada del test set. La evaluación usa hasta `40` ejemplos por resultado de matriz de confusión (`TN`, `FP`, `FN`, `TP`) en cada dataset, incluyendo todos los errores disponibles cuando son menos que ese máximo. En total se evaluaron `260` muestras únicas: `132` de Kepler y `128` de TESS.

## Sobre Quantus

**Quantus no se usó literalmente como dependencia de Python.** La razón práctica fue mantener el repositorio reproducible con las dependencias ya incluidas y evitar agregar una instalación nueva solo para una métrica. Sin embargo, la métrica implementada sigue el mismo objetivo que la Faithfulness Correlation sugerida en la literatura y disponible en Quantus: perturbar regiones de la entrada, medir el cambio en la salida del modelo y correlacionar ese cambio con la importancia asignada por el método XAI.

Por lo tanto, en el informe conviene escribirlo como: *"Se calculó una métrica de fidelidad tipo Faithfulness Correlation, equivalente en objetivo a la implementación disponible en Quantus, pero implementada localmente para mantener reproducibilidad del pipeline"*. No conviene afirmar que se usó Quantus si no se importó la biblioteca.

## Sobre SHAP

SHAP no fue usado y se puede eliminar del paper. Para estas curvas de luz, SHAP por punto temporal puede ser costoso y difícil de interpretar, especialmente en Kepler con 2001 puntos globales y 201 puntos locales. La comparación principal queda mejor defendida con Saliency, Integrated Gradients, Occlusion, Consensus, MC Dropout y métricas de fidelidad.

## Protocolo ampliado

Para cada muestra seleccionada se explicó la clase predicha por el modelo, no necesariamente la etiqueta verdadera. Esto evalúa la fidelidad de la explicación respecto de la decisión real del modelo. Luego se generaron `32` perturbaciones aleatorias por muestra, apagando el `10%` de los puntos de la vista global y el `10%` de los puntos de la vista local. Para cada perturbación se midió la caída de confianza hacia la clase predicha. Finalmente, se calculó la correlación entre:

- la suma de atribuciones XAI en los puntos perturbados;
- la caída de confianza del modelo después de perturbar esos puntos.

Un valor positivo indica que las regiones marcadas como importantes tienden a producir mayor cambio en la salida cuando se intervienen. Un valor cercano a cero indica fidelidad débil. Un valor negativo indica que, bajo este protocolo, el mapa puede estar resaltando regiones que no explican bien la salida del modelo.

## Distribución de muestras evaluadas

| dataset | TN | FP | FN | TP |
| --- | --- | --- | --- | --- |
| kepler | 40 | 35 | 17 | 40 |
| tess | 40 | 28 | 20 | 40 |

## Resumen de resultados ampliados

| dataset | method | n_unique_samples | valid_n | pearson_mean | spearman_mean | mean_confidence_drop | negative_drop_fraction_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| kepler | Integrated Gradients | 132 | 123 | 0.1659 | 0.1668 | 0.0402 | 0.2704 |
| kepler | Consensus | 132 | 123 | 0.1598 | 0.1478 | 0.0402 | 0.2704 |
| kepler | Saliency | 132 | 123 | 0.1267 | 0.1454 | 0.0402 | 0.2704 |
| kepler | Occlusion | 132 | 123 | 0.1255 | 0.1049 | 0.0402 | 0.2704 |
| tess | Occlusion | 128 | 128 | 0.5390 | 0.5544 | 0.0678 | 0.3340 |
| tess | Consensus | 128 | 128 | 0.4013 | 0.4081 | 0.0678 | 0.3340 |
| tess | Integrated Gradients | 128 | 128 | 0.2879 | 0.2960 | 0.0678 | 0.3340 |
| tess | Saliency | 128 | 128 | 0.1063 | 0.0999 | 0.0678 | 0.3340 |

## Interpretación

La evaluación ampliada mejora la defensa del trabajo porque ya no depende solo de los 6 ejemplos usados para figuras individuales. Ahora la fidelidad se mide sobre una muestra estratificada que incluye verdaderos negativos, falsos positivos, falsos negativos y verdaderos positivos.

La conclusión principal debe expresarse con cautela: las explicaciones tienen fidelidad parcial, no absoluta. Si los valores son positivos, hay evidencia de que las regiones importantes están relacionadas con cambios reales en la salida del modelo. Si algunos métodos quedan cerca de cero o negativos, eso indica que esos mapas deben usarse como visualización complementaria y no como prueba causal.

En general, **Occlusion** y **Consensus** son los métodos más defendibles para interpretar el modelo, porque conectan mejor la importancia visual con cambios de confianza. Integrated Gradients también es útil, pero puede variar más por muestra. Saliency es barato y sirve como diagnóstico rápido, aunque puede ser más ruidoso.

## Limitaciones

Aunque esta evaluación es más fuerte que la inicial, sigue teniendo limitaciones. Primero, no se evaluó todo el test set, sino una muestra estratificada ampliada para mantener tiempo de cómputo razonable. Segundo, las perturbaciones con ceros pueden generar curvas fuera de distribución. Tercero, el modelo es una red neuronal convolucional no lineal con dos ramas, por lo que las interacciones entre vista global y local no se capturan completamente con una correlación local.

Esto es importante para responder a la observación de Miró-Nicolau et al. (2025): las métricas de fidelidad son más confiables en modelos lineales que en redes neuronales profundas. Por eso estos valores deben interpretarse como evidencia comparativa entre métodos XAI bajo un protocolo fijo, no como certificación causal absoluta.

## Figuras generadas

- `faithfulness_correlation_ampliada_por_metodo.png`: compara Pearson y Spearman medios por método y dataset.
- `faithfulness_correlation_ampliada_distribucion.png`: muestra la distribución por muestra de Pearson para cada método.
- `faithfulness_occlusion_ampliada_por_resultado.png`: resume el comportamiento de Occlusion por resultado de matriz de confusión.

## Cómo reportarlo en el paper

Texto recomendado:

"Para evaluar si las explicaciones reflejan el comportamiento del modelo, se calculó una métrica tipo Faithfulness Correlation sobre una muestra estratificada del conjunto de test. Aunque Quantus no se utilizó literalmente como dependencia, el protocolo implementa la misma idea: perturbar regiones de la entrada y correlacionar la importancia asignada por XAI con la caída de confianza del modelo. Esta evaluación se realizó sobre una muestra ampliada que incluye verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos. Los resultados indican fidelidad parcial y dependiente del método, siendo Occlusion y Consensus las alternativas más defendibles. Debido a la naturaleza no lineal de la red neuronal, estos valores se interpretan como evidencia comparativa y no como prueba causal absoluta."
