import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Predicción de Vinos")

st.title("Predicción de Éxito de Vinos Tintos")
st.write("Esta aplicación predice el nivel de éxito de un vino tinto según sus características químicas.")

# =========================
# Cargar y limpiar datos
# =========================
df = pd.read_csv("vinos_tintos.csv")
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

# =========================
# SIDEBAR
# =========================
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

# =========================
# PREDICCIÓN CON COLORES
# =========================
if st.sidebar.button("Predecir"):
    datos = np.array([[acidez_fija, acidez_volatil, acido_citrico, azucar_residual,
                       cloruros, dioxido_libre, dioxido_total, densidad, ph, sulfatos, alcohol]])
    
    prediccion = modelo.predict(datos)[0]
    
    st.subheader("Resultado de la predicción")
    
    if prediccion >= 65:
        st.success(f"Success predicho: **{prediccion:.2f}**")
        st.markdown("<h3 style='color:green;'>Nivel de éxito: ALTO</h3>", unsafe_allow_html=True)
        st.write("Este vino tiene un alto potencial de éxito.")
        
    elif prediccion >= 50:
        st.info(f"Success predicho: **{prediccion:.2f}**")
        st.markdown("<h3 style='color:orange;'>Nivel de éxito: MEDIO</h3>", unsafe_allow_html=True)
        st.write("Este vino tiene un nivel de éxito intermedio.")
        
    else:
        st.error(f"Success predicho: **{prediccion:.2f}**")
        st.markdown("<h3 style='color:red;'>Nivel de éxito: BAJO</h3>", unsafe_allow_html=True)
        st.write("Este vino presenta un nivel de éxito bajo.")

# =========================
# RESULTADOS DEL MODELO
# =========================
st.subheader("Resultados del modelo Random Forest")

y_pred = modelo.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.write(f"**R²:** {r2:.4f}")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")

# =========================
# Clasificación
# =========================
st.subheader("Clasificación del Success")
st.markdown("""
- <span style='color:green; font-weight:bold;'>ALTO:</span> 65 o más  
- <span style='color:orange; font-weight:bold;'>MEDIO:</span> entre 50 y 64  
- <span style='color:red; font-weight:bold;'>BAJO:</span> menos de 50  
""", unsafe_allow_html=True)

# =========================
# GRÁFICO 2
# =========================
st.subheader("Valores reales vs predichos")

fig2, ax2 = plt.subplots()
ax2.scatter(y_test, y_pred, alpha=0.6, color="purple")
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
ax2.set_xlabel("Valores reales")
ax2.set_ylabel("Valores predichos")
ax2.set_title("Comparación entre valores reales y predichos")
st.pyplot(fig2)
