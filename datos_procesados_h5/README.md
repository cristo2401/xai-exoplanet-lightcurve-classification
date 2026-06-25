# Datos procesados HDF5

Esta carpeta se genera localmente desde los TFRecords incluidos en el repositorio.

No se incluyen archivos `.h5` en GitHub porque son artefactos generados y pueden superar limites practicos de tamano de archivo.

Para regenerarlos desde la raiz del repositorio:

```bash
python codigo/convertir_kepler_tfrecord_a_h5.py --output_h5 datos_procesados_h5/kepler_dataset.h5
python codigo/convertir_tess_tfrecord_a_h5.py --mode pc_only --output_h5 datos_procesados_h5/tess_dataset.h5
python codigo/convertir_tess_tfrecord_a_h5.py --mode triage --output_h5 datos_procesados_h5/tess_dataset_triage.h5
```
