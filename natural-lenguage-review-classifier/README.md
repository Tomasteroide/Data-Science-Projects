# 🎬 Natural Language Review Classifier

Clasificador de sentimiento basado en reseñas textuales de películas utilizando técnicas de Procesamiento de Lenguaje Natural (NLP) y modelos de clasificación supervisada. Este proyecto tiene como objetivo identificar automáticamente si una reseña es positiva o negativa, permitiendo su uso en plataformas de streaming, foros o sitios de reseñas.

---

## 🎯 Objetivo

Desarrollar un modelo de clasificación binaria que determine el sentimiento de una reseña de película (positiva o negativa), utilizando un conjunto de datos etiquetado de IMDb y alcanzando al menos un F1-score de 0.85.

---

## 🧰 Tecnologías y Librerías

- Python  
- Pandas, NumPy  
- Scikit-learn  
- NLTK  
- Matplotlib, Seaborn  
- TF-IDF  
- Regresión Logística

---

## ⚙️ Flujo del Proyecto

1. **Carga y análisis exploratorio de datos**  
   - Revisión del dataset de IMDb  
   - Análisis de balance de clases y longitud de reseñas

2. **Preprocesamiento de texto**  
   - Conversión a minúsculas, limpieza de símbolos  
   - Eliminación de stopwords, tokenización y lematización

3. **Vectorización de texto**  
   - Representación numérica de textos mediante TF-IDF

4. **Entrenamiento de modelos**  
   - Modelos evaluados: Regresión Logística, Gradient Boosting y Árbol de Decisión  
   - Evaluación con accuracy, precision, recall y F1-score

5. **Clasificación de reseñas personalizadas**  
   - Clasificación de textos ingresados por el usuario  
   - Análisis de palabras más influyentes en la predicción

---

## 📊 Resultados

- **Mejor modelo:** Regresión Logística con TF-IDF  
- **F1-score:** 0.88  
- El modelo supera el umbral objetivo y permite clasificar correctamente el sentimiento de reseñas de texto en tiempo real.

---

