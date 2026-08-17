import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Predicción de Vinos", page_icon="🍷", layout="centered")

st.title("🍷 Predicción de Éxito de Vinos Tintos")
st.write("Ingresa las características químicas del vino para predecir su nivel de éxito.")

# =========================
# Cargar datos y entrenar modelo
# =========================
@st.cache_data
def cargar_datos():
    df = pd.read_csv("vinos_tintos.csv")
    df = df.drop_duplicates()
    df = df.dropna()
    df["country"] = df["country"].replace({"Spagna": "Spain", "Espana": "Spain"})
    return df

df = cargar_datos()

@st.cache_resource
def entrenar_modelo():
    X = df.drop(columns=["success", "country", "pricing"])
    y = df["success"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = RandomForestRegressor(n_estimators=150, random_state=42)
    modelo.fit(X_train, y_train)
    return modelo

modelo_rf = entrenar_modelo()

# =========================
# Sidebar - Entradas
# =========================
st.sidebar.header("Características del Vino")

# Valores por defecto (promedio aproximado)
fixed_acidity = st.sidebar.slider("Acidez Fija", 4.5, 16.0, 8.3, 0.1)
volatile_acidity = st.sidebar.slider("Acidez Volátil", 0.10, 1.60, 0.53, 0.01)
citric_acid = st.sidebar.slider("Ácido Cítrico", 0.0, 1.0, 0.27, 0.01)
residual_sugar = st.sidebar.slider("Azúcar Residual", 0.9, 15.5, 2.5, 0.1)
chlorides = st.sidebar.slider("Cloruros", 0.01, 0.60, 0.08, 0.001)
free_sulfur_dioxide = st.sidebar.slider("Dióxido de Azufre Libre", 1.0, 70.0, 16.0, 1.0)
total_sulfur_dioxide = st.sidebar.slider("Dióxido de Azufre Total", 6.0, 290.0, 46.0, 1.0)
density = st.sidebar.slider("Densidad", 0.990, 1.004, 0.997, 0.0001)
pH = st.sidebar.slider("pH", 2.7, 4.0, 3.3, 0.01)
sulphates = st.sidebar.slider("Sulfatos", 0.3, 2.0, 0.65, 0.01)
alcohol = st.sidebar.slider("Alcohol", 0.08, 0.15, 0.10, 0.001)

# =========================
# Botones de ejemplo
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("Ejemplos rápidos")

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("Success Alto", use_container_width=True):
        st.session_state.ejemplo = "alto"

with col2:
    if st.button("Success Bajo", use_container_width=True):
        st.session_state.ejemplo = "bajo"

# Aplicar ejemplos
if "ejemplo" in st.session_state:
    if st.session_state.ejemplo == "alto":
        fixed_acidity, volatile_acidity, citric_acid = 11.0, 0.25, 0.50
        residual_sugar, chlorides = 2.5, 0.06
        free_sulfur_dioxide, total_sulfur_dioxide = 30.0, 100.0
        density, pH, sulphates, alcohol = 0.998, 3.2, 0.90, 0.120
    elif st.session_state.ejemplo == "bajo":
        fixed_acidity, volatile_acidity, citric_acid = 5.0, 1.30, 0.00
        residual_sugar, chlorides = 1.5, 0.15
        free_sulfur_dioxide, total_sulfur_dioxide = 5.0, 20.0
        density, pH, sulphates, alcohol = 0.995, 3.7, 0.40, 0.085

# =========================
# Predicción
# =========================
if st.sidebar.button("Predecir Éxito", type="primary", use_container_width=True):
    datos = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                       chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                       density, pH, sulphates, alcohol]])
    
    prediccion = modelo_rf.predict(datos)[0]
    
    st.markdown("---")
    
    if prediccion >= 65:
        st.success(f"### 🍷 Success predicho: **{prediccion:.2f}** (Alto)")
    elif prediccion >= 50:
        st.info(f"### 🍷 Success predicho: **{prediccion:.2f}** (Medio)")
    else:
        st.error(f"### 🍷 Success predicho: **{prediccion:.2f}** (Bajo)")
    
    st.caption("El modelo Random Forest analiza las características químicas para estimar el nivel de éxito del vino.")
