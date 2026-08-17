import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Predicción de Vinos")

st.title("Predicción de Éxito de Vinos Tintos")
st.write("Esta aplicación predice el nivel de éxito de un vino tinto según sus características químicas.")

# Cargar el dataset
df = pd.read_csv("vinos_tintos.csv")

# Limpieza básica
df = df.drop_duplicates()
df = df.dropna()
df["country"] = df["country"].replace({"Spagna": "Spain", "Espana": "Spain"})

# Separar variables
X = df.drop(columns=["success", "country", "pricing"])
y = df["success"]

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Sidebar
st.sidebar.header("Ingrese las características del vino")

acidez_fija = st.sidebar.number_input("Acidez Fija", value=7.4)
acidez_volatil = st.sidebar.number_input("Acidez Volátil", value=0.70)
acido_citrico = st.sidebar.number_input("Ácido Cítrico", value=0.00)
azucar_residual = st.sidebar.number_input("Azúcar Residual", value=1.9)
cloruros = st.sidebar.number_input("Cloruros", value=0.076)
dioxido_libre = st.sidebar.number_input("Dióxido de Azufre Libre", value=11.0)
dioxido_total = st.sidebar.number_input("Dióxido de Azufre Total", value=34.0)
densidad = st.sidebar.number_input("Densidad", value=0.9978)
ph = st.sidebar.number_input("pH", value=3.51)
sulfatos = st.sidebar.number_input("Sulfatos", value=0.56)
alcohol = st.sidebar.number_input("Alcohol", value=0.094)

# Botón para predecir
if st.sidebar.button("Predecir"):
    datos = np.array([[acidez_fija, acidez_volatil, acido_citrico, azucar_residual,
                       cloruros, dioxido_libre, dioxido_total, densidad, ph, sulfatos, alcohol]])
    
    prediccion = modelo.predict(datos)[0]
    
    # Mensajes según el resultado
    if prediccion >= 65:
        st.success(f"Success predicho: {prediccion:.2f}")
        st.write("Este vino tiene un nivel de éxito **alto**.")
    elif prediccion >= 50:
        st.info(f"Success predicho: {prediccion:.2f}")
        st.write("Este vino tiene un nivel de éxito **medio**.")
    else:
        st.error(f"Success predicho: {prediccion:.2f}")
        st.write("Este vino tiene un nivel de éxito **bajo**.")
