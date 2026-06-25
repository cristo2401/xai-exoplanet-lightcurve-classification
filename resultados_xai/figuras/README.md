# Figuras XAI generadas

Esta carpeta se regenera ejecutando:

```bash
python xai_mejores_modelos.py --dataset both --mc-samples 100 --smooth-samples 32 --ig-steps 64 --max-examples 6 --batch-size 512
```

En el ZIP liviano no se incluyen todas las figuras XAI generadas para evitar duplicar artefactos pesados. Las figuras principales listas para informe se conservan en `notebook/resultados/figuras_academicas/`.
