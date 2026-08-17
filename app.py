import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Predicción de Vinos", page_icon="")

st.title("Predicción de Success de Vinos Tintos")
st.write("Ingresa las características químicas del vino para predecir su nivel de éxito.")

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
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    return modelo

modelo_rf = entrenar_modelo()

st.sidebar.header("Características del Vino")

fixed_acidity = st.sidebar.number_input("Fixed Acidity", value=7.4)
volatile_acidity = st.sidebar.number_input("Volatile Acidity", value=0.70)
citric_acid = st.sidebar.number_input("Citric Acid", value=0.00)
residual_sugar = st.sidebar.number_input("Residual Sugar", value=1.9)
chlorides = st.sidebar.number_input("Chlorides", value=0.076)
free_sulfur_dioxide = st.sidebar.number_input("Free Sulfur Dioxide", value=11.0)
total_sulfur_dioxide = st.sidebar.number_input("Total Sulfur Dioxide", value=34.0)
density = st.sidebar.number_input("Density", value=0.9978)
pH = st.sidebar.number_input("pH", value=3.51)
sulphates = st.sidebar.number_input("Sulphates", value=0.56)
alcohol = st.sidebar.number_input("Alcohol", value=0.094)

if st.sidebar.button("Predecir Success"):
    datos = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                       chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                       density, pH, sulphates, alcohol]])
    
    prediccion = modelo_rf.predict(datos)[0]
    st.success(f"**Success predicho: {prediccion:.2f}**")
