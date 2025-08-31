# Prompt v1

Genera un **script en Python bien documentado** que implemente el **fine-tuning del modelo RoBERTa** para análisis de sentimientos en reseñas de texto.

El script debe estar dividido en secciones claras con comentarios y debe incluir todo el flujo de trabajo, en este orden:

1. **Importaciones necesarias**
   - PyTorch, Hugging Face Transformers, scikit-learn, datasets, etc.

2. **Carga y exploración de datos**
   - Dataset de reseñas con etiquetas `positive` y `negative`.
   - Conversión de etiquetas a valores numéricos.
   - División en train y test (80/20).

3. **Tokenización con RoBERTa**
   - Uso de `RobertaTokenizer` (`roberta-base`).
   - Configuración de `MAX_LEN`.
   - Conversión de textos a tensores (`input_ids`, `attention_mask`, `token_type_ids`).

4. **Creación de dataset y dataloader**
   - Implementación de la clase `SentimentData`.
   - Configuración de `DataLoader` para entrenamiento y validación.

5. **Definición del modelo**
   - Clase `RobertaClass` que usa `RobertaModel.from_pretrained("roberta-base")`.
   - Capas adicionales: lineal intermedia (768→768), ReLU, dropout (0.2), salida lineal (768→2).

6. **Configuración de entrenamiento**
   - Función de pérdida: `CrossEntropyLoss`.
   - Optimizador: `Adam` con `learning_rate=1e-5`.
   - Funciones de entrenamiento (`train`) y validación (`valid`) con métricas de precisión y f1.

7. **Entrenamiento del modelo**
   - Ejecutar el fine-tuning por 1 época.
   - Mostrar métricas intermedias y finales.

8. **Evaluación final**
   - Calcular `accuracy`, `precision`, `recall`, `f1-score` sobre el test set.
   - Mostrar `classification_report`.

9. **Guardado y carga del modelo entrenado**
   - Guardar el modelo y el tokenizer.
   - Función `predict(text)` para clasificar nuevas reseñas.

10. **Ejemplo de uso**
    - Pasar una lista de reseñas nuevas al modelo y mostrar si son `positive` o `negative`.

🔹 El código debe estar organizado, modular y comentado paso a paso para que cualquier persona pueda entenderlo y ejecutarlo en Google Colab o localmente.

