# RoBERTa-fine-tuning-transformers

Este trabajo presenta el **proceso de fine-tuning del modelo RoBERTa** para la tarea de **análisis de sentimientos en reseñas** (reviews). El objetivo principal fue construir un **clasificador automático** capaz de identificar si una opinión corresponde a un sentimiento **positivo** o **negativo**.  

## Flujo de trabajo

1. **Preparación de datos**  
   - Se utilizó un dataset de reseñas etiquetadas en positivo y negativo.  
   - Las etiquetas fueron codificadas numéricamente.  
   - Se construyeron datasets de entrenamiento (80%) y prueba (20%).  
   - Se implementó una clase `SentimentData` para manejar los datos y preparar los *batches* mediante `DataLoader`.

2. **Modelado con RoBERTa**  
   - Se empleó el modelo **`roberta-base`** de Hugging Face, adaptado a una salida binaria.  
   - Se definieron capas adicionales: una lineal intermedia, dropout como regularizador y una capa final de clasificación.  
   - El entrenamiento se realizó con **CrossEntropyLoss** y el optimizador **Adam**.  

3. **Entrenamiento y validación**  
   - Se ejecutó el proceso de fine-tuning en GPU.  
   - Se midieron métricas de desempeño como **accuracy** y **f1-score**.  
   - El entrenamiento requirió aproximadamente 25 minutos para 1 época.  

4. **Resultados**  
   - El modelo alcanzó un **f1-score de 0.91** sobre el conjunto de prueba, lo que refleja un **alto desempeño en la clasificación de sentimientos**.  
   - Los reportes de clasificación muestran un balance adecuado entre precisión y recall en ambas clases.  

5. **Despliegue y uso**  
   - Se guardó el modelo fine-tuneado para reutilización.  
   - Se implementó una función `predict` que permite clasificar nuevas reseñas de manera automática.  
   - El modelo fue evaluado sobre un conjunto de datos externo, confirmando su capacidad de generalización.  

## Conclusiones
El trabajo demuestra que **RoBERTa, con un fine-tuning adecuado, ofrece un rendimiento sobresaliente para análisis de sentimientos** en texto. La integración con la librería **Hugging Face** simplifica la construcción de *pipelines* de entrenamiento y predicción, haciendo viable su aplicación en entornos reales como sistemas de recomendación, análisis de redes sociales o atención al cliente.  
