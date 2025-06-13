import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os
from scipy.special import lambertw

# --- DEFINICIÓN DE LA FUNCIÓN Y CONSTANTES DEL MODELO ---
l = 0.64
epsilon = 1445 + 35
el = l * epsilon

def model_4(t, c1, k, x1, y1):
    return el * (x1 - y1 * np.real(lambertw(c1 * np.exp(-k * t))))

# --- CONFIGURACIÓN DE LA PÁGINA DE STREAMLIT ---
st.set_page_config(
    page_title="Monitoreo de Nanopartículas en Tiempo Real",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TÍTULO Y DESCRIPCIÓN ---
st.title("⚗️ Monitoreo de Formación de Nanopartículas")
st.markdown("Esta aplicación simula en tiempo real la formación de nanopartículas, añadiendo un nuevo punto de datos a la gráfica y a la tabla cada 5 segundos.")

# --- CARGA DEL MODELO (CACHEADO) ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "files", "models", "model_4.pkl")

@st.cache_resource
def load_model(path):
    try:
        with open(path, 'rb') as file:
            model_data = pickle.load(file)
        return model_data['model'], model_data['parameters']
    except FileNotFoundError:
        st.error(f"Error Crítico: No se encontró el archivo del modelo en '{path}'. Asegúrate de que el archivo existe.")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.stop()

model, params = load_model(MODEL_PATH)

# --- FUNCIONES DE SIMULACIÓN ---
def model_4_prediction(t_values, c1, k, x1, y1):
    return el * (x1 - y1 * np.real(lambertw(c1 * np.exp(-k * t_values))))

def simulate_data_point(current_sim_time_seconds, model_func, model_params, noise_level=0.01):
    time_min = current_sim_time_seconds / 60.0
    predicted_absorbance = model_func(np.array([time_min]), *model_params)[0]
    noise = np.random.normal(0, noise_level * predicted_absorbance)
    return time_min, predicted_absorbance + noise

# --- ESTADO DE LA SESIÓN ---
if 'running' not in st.session_state:
    st.session_state.running = False
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["Tiempo (min)", "Absorbancia (u.a.)"])
if 'sim_time' not in st.session_state:
    st.session_state.sim_time = 0
if 'last_update' not in st.session_state:
    st.session_state.last_update = 0

# --- CONTROLES DE LA BARRA LATERAL ---
st.sidebar.header("🕹️ Control de Simulación")
if st.sidebar.button("▶️ Iniciar Simulación", type="primary"):
    st.session_state.running = True
    st.session_state.data = pd.DataFrame(columns=["Tiempo (min)", "Absorbancia (u.a.)"])
    st.session_state.sim_time = 0
    st.session_state.last_update = time.time()
    st.rerun()

if st.sidebar.button("⏹️ Detener Simulación"):
    st.session_state.running = False
    st.sidebar.success("Simulación detenida.")
    st.rerun()

# --- LAYOUT DE LA APLICACIÓN ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Gráfica en Tiempo Real")
    chart_placeholder = st.empty()

with col2:
    st.subheader("📋 Tabla de Registros")
    table_placeholder = st.empty()

# --- LÓGICA DE SIMULACIÓN Y ACTUALIZACIÓN ---
# Rango fijo para el eje Y
Y_AXIS_RANGE = [0.0, 1.0]

if not st.session_state.running:
    st.info("Presiona 'Iniciar Simulación' para comenzar.")
    if not st.session_state.data.empty:
        df_display = st.session_state.data.set_index("Tiempo (min)")
        # CAMBIO CLAVE: Añadido el parámetro 'y'
        chart_placeholder.line_chart(df_display, color="#ffca3a", y=Y_AXIS_RANGE, use_container_width=True)
        table_placeholder.dataframe(df_display.style.format("{:.4f}"), use_container_width=True)

if st.session_state.running:
    st.sidebar.info("Simulación en curso...")
    max_sim_time_minutes = 180
    max_sim_time_seconds = max_sim_time_minutes * 60

    df_display = st.session_state.data.set_index("Tiempo (min)")
    # CAMBIO CLAVE: Añadido el parámetro 'y'
    chart_placeholder.line_chart(df_display, color="#ffca3a", y=Y_AXIS_RANGE, use_container_width=True)
    table_placeholder.dataframe(df_display.style.format({"Absorbancia (u.a.)": "{:.4f}"}), use_container_width=True)

    if time.time() - st.session_state.last_update > 5:
        if st.session_state.sim_time <= max_sim_time_seconds:
            time_min, new_abs = simulate_data_point(st.session_state.sim_time, model_4_prediction, params)
            new_row = pd.DataFrame([{"Tiempo (min)": time_min, "Absorbancia (u.a.)": new_abs}])
            st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
            st.session_state.sim_time += 5
            st.session_state.last_update = time.time()
        else:
            st.session_state.running = False
            st.sidebar.success("Simulación completada.")
    
    if st.session_state.running:
        time.sleep(1)
        st.rerun()