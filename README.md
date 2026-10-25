# Pronóstico del consumo eléctrico diario de un hogar con redes recurrentes

Proyecto integrador de Técnicas avanzadas de modelado de IA, Unidad 4, actividad 2.

Integrantes: Saud Mauricio González Rodríguez.

## Contenido

`pipeline_avanzado.ipynb` es el notebook ejecutado, con todas sus salidas. `pipeline_avanzado.py` contiene el mismo código en formato de script, y `a_notebook.py` lo convierte en notebook. La carpeta `resultados` guarda las tablas de experimentos, las curvas, la comparación con la línea base, las pruebas estadísticas, el análisis de errores, las figuras y los pesos de un modelo final. `mlruns.zip` contiene los registros de MLflow y `environment.yml` describe el entorno usado (conda forge, PyTorch para CPU).

## Datos

Se usa el dataset Individual Household Electric Power Consumption de UCI (DOI 10.24432/C58K54, licencia CC BY 4.0). El archivo no se incluye por su tamaño. Descárguelo e indique su ruta con la variable de entorno `RUTA_DATOS`.

## Ejecución

```
conda env create -f environment.yml
conda activate u4a2
RUTA_DATOS=/ruta/household_power_consumption.txt jupyter nbconvert --to notebook --execute pipeline_avanzado.ipynb
```

Por defecto, el notebook lee los resultados guardados de la búsqueda y vuelve a entrenar solo los cinco modelos finales. Para repetir la búsqueda completa, que tarda unas 3 horas en CPU, use `EJECUTAR_BUSQUEDA=1`. Para ver los registros de MLflow, descomprima `mlruns.zip` y ejecute `MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri mlruns`.
