# 📉 Customer Churn Prediction

Este proyecto tiene como objetivo predecir qué clientes de una empresa de telecomunicaciones tienen mayor probabilidad de cancelar sus servicios, permitiendo a la organización implementar estrategias proactivas de retención. Se utilizó un enfoque de aprendizaje supervisado, combinado con técnicas de balanceo de clases, ingeniería de características y múltiples algoritmos de clasificación.

---

## 🎯 Objetivo

Construir un modelo de clasificación binaria que identifique clientes propensos a cancelar su contrato, utilizando variables transaccionales, de comportamiento y características personales.

---

## 🧰 Tecnologías y Librerías

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost, LightGBM, CatBoost
- SMOTETomek, SMOTEENN
- AdaBoost, Random Forest, Gradient Boosting
- GridSearchCV
- Matplotlib, Seaborn

---

## ⚙️ Flujo del proyecto

1. **Carga y exploración de datos**
   - Integración de múltiples fuentes: clientes, historial, servicios, tarifas y target
   - Análisis de distribución, nulos, correlaciones y creación de variables nuevas

2. **Preprocesamiento**
   - Unificación y limpieza de datos
   - Generación de nuevas variables como duración del contrato, proporciones de cargos, etc.
   - Codificación de variables categóricas y escalamiento

3. **Balanceo de clases**
   - Se probaron métodos como SMOTETomek y SMOTEENN para corregir el desbalance natural entre clientes que cancelan y los que no

4. **Modelado**
   - Entrenamiento de múltiples modelos: Decision Tree, Random Forest, Logistic Regression, AdaBoost, XGBoost, LightGBM y CatBoost
   - Ajuste de hiperparámetros con GridSearchCV
   - Evaluación con métricas: F1-score, accuracy, ROC-AUC, matriz de confusión

5. **Optimización de thresholds**
   - Ajuste del umbral de probabilidad para maximizar recall sin sacrificar precisión

---

## 📊 Resultados

- **Mejores modelos:** AdaBoost y Gradient Boosting + SMOTEENN  
- **F1-score:** 1.00  
- **ROC-AUC:** 1.00  
- **Recall:** 1.00  
- Los modelos permiten identificar con alta precisión a los clientes en riesgo de fuga, facilitando campañas dirigidas de retención.


