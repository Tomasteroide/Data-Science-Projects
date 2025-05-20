# 🚕 Taxi Demand Prediction

Este proyecto tiene como objetivo predecir el número de órdenes de taxi por hora en aeropuertos, utilizando datos históricos de Sweet Lift Taxi. El modelo busca anticipar picos de demanda para facilitar la planificación operativa y atraer más conductores en momentos críticos.

---

## 🎯 Objetivo

Construir un modelo de regresión que prediga la cantidad de pedidos de taxi por hora, alcanzando un error cuadrático medio (RMSE) inferior a 48.

---

## 🧰 Tecnologías y Librerías

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Random Forest, Decision Tree, Linear Regression  
- Matplotlib, Seaborn

---

## ⚙️ Flujo del proyecto

1. **Carga y preparación de datos**
   - Conversión de fechas, resampleo por hora, agregación y manejo de series temporales

2. **Análisis exploratorio**
   - Visualización de tendencias, estacionalidad y fluctuaciones horarias

3. **Ingeniería de características**
   - Variables temporales (hora, día, mes, día de la semana)
   - Lags y medias móviles para capturar patrones históricos

4. **Modelado y evaluación**
   - Comparación de modelos: regresión lineal, árbol de decisión y random forest
   - Métrica principal: RMSE en conjunto de prueba

---

## 📊 Resultados

- **Mejor modelo:** Random Forest  
- **RMSE obtenido:** 46.39  
- El modelo cumple con el objetivo del negocio y permite anticipar con precisión la demanda por hora.

