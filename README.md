# Series Temporales

Este proyecto se centra en predecir el valor de mercado de coches usados utilizando técnicas de series temporales. El análisis se basa en datos históricos de precios y especificaciones técnicas de los vehículos para generar un modelo predictivo que ayude a identificar tendencias y patrones de precios futuros.

## Tecnologías utilizadas
- Python
- pandas
- scikit-learn
- LightGBM
- Matplotlib

## Objetivo
Desarrollar un modelo de machine learning basado en series temporales para predecir el valor futuro de coches usados en el mercado, utilizando datos históricos de precios y especificaciones técnicas.

## Contexto
El mercado de coches usados es altamente variable, con precios que fluctúan en función de factores como la demanda, la marca, el modelo, y las especificaciones técnicas. Este proyecto busca aplicar técnicas de series temporales para predecir el valor de los vehículos a lo largo del tiempo, proporcionando una herramienta valiosa para concesionarios y compradores.

## Descripción del Proyecto
El proyecto involucra el uso de datos de coches usados, incluidas sus especificaciones técnicas y precios históricos, para desarrollar un modelo de series temporales que prediga el valor futuro de estos vehículos.

Pasos del proyecto:
1. **Exploración de datos**: Se inspeccionaron las tendencias y patrones en los datos históricos de precios de coches usados.
2. **Preprocesamiento de datos**: Se limpió el conjunto de datos, se manejaron valores faltantes y se transformaron las variables de interés.
3. **Entrenamiento del modelo**: Se entrenaron varios modelos de series temporales utilizando algoritmos como LightGBM.
4. **Evaluación del modelo**: Se utilizaron métricas como MAE (Mean Absolute Error) y RMSE (Root Mean Squared Error) para medir el rendimiento del modelo.

## Proceso

### Exploración de Datos
Se realizó un análisis exploratorio utilizando pandas y matplotlib para identificar las tendencias y patrones en los precios de los coches usados a lo largo del tiempo. Se observaron fluctuaciones de precios en función de la antigüedad, el kilometraje y el tipo de vehículo.

### Preprocesamiento
Se transformaron los datos utilizando técnicas de normalización y manejo de valores faltantes. Además, se crearon nuevas variables, como la tasa de depreciación, para mejorar las predicciones.

### Desarrollo de Modelos
Se entrenaron modelos de series temporales, siendo LightGBM el modelo más efectivo debido a su capacidad para manejar datos tabulares y generar predicciones precisas en series temporales.

### Evaluación del Modelo
El modelo fue evaluado utilizando métricas como:
- **MAE (Mean Absolute Error)**: Se utilizó para medir el error promedio entre los valores predichos y los reales.
- **RMSE (Root Mean Squared Error)**: El error cuadrático medio indicó un buen ajuste del modelo, con una baja variabilidad en las predicciones.

## Resultados
El modelo de **LightGBM** mostró un excelente rendimiento, con una baja tasa de error en la predicción del valor de mercado de los coches usados. Esto proporciona a concesionarios y compradores una herramienta valiosa para prever el precio futuro de los vehículos y tomar decisiones de compra más informadas.

## Conclusiones
El proyecto demostró que las técnicas de series temporales y los modelos de machine learning pueden ser utilizados para predecir el valor futuro de coches usados con alta precisión. Esto puede ayudar a los concesionarios a ajustar mejor sus precios y a los compradores a realizar compras más inteligentes.

### Futuras mejoras
- Incluir más características en el análisis, como datos macroeconómicos que influyen en el mercado de coches usados.
- Probar otros modelos de series temporales como Prophet o ARIMA para comparar el rendimiento.
- Implementar técnicas de optimización de hiperparámetros para mejorar el modelo de LightGBM.

### Enlace al proyecto
[Series Temporales](https://github.com/ErayFaSol/Sprint-13-Series-temporales)
