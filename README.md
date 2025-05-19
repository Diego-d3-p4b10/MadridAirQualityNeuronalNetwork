# 🌫️ Madrid Air Quality Neuronal Network

Este proyecto aplica una red neuronal LSTM implementada en PyTorch para predecir los niveles de contaminación del aire en Madrid. Se estructura en dos fases principales: preparación del dataset y entrenamiento del modelo.

---

## 🧠 Objetivo

Desarrollar un modelo predictivo basado en aprendizaje profundo que, a partir de datos históricos sobre la calidad del aire y condiciones meteorológicas, anticipe los niveles de contaminantes en la ciudad de Madrid.

---

## 🗂️ Estructura del Proyecto

- `DATA.ipynb`: Notebook interactivo para carga, limpieza, normalización y estructuración de los datos.
- `Entrenamiento.py`: Script que construye, entrena y guarda una red LSTM en PyTorch.
- `README.md`: Este archivo, con toda la documentación necesaria para ejecutar el proyecto.

---

## 📥 Datos Utilizados

Los datos provienen de estaciones de medición de calidad del aire de la ciudad de Madrid, e incluyen:

- **Contaminantes**: NO₂, PM10, O₃, entre otros.
- **Variables climáticas**: temperatura, humedad, velocidad del viento.
- **Tiempo**: timestamp para análisis temporal por hora/día.

Estos datos son tratados y normalizados en `DATA.ipynb` para que puedan ser consumidos por la red neuronal.

---

## ⚙️ Requisitos del Proyecto

Instala las dependencias necesarias ejecutando:

pip install torch pandas numpy matplotlib scikit-learn jupyter
