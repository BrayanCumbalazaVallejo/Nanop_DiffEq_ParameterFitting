import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os
from scipy.special import lambertw

# --- DEFINICIÓN DE LA FUNCIÓN MODEL_4 Y SUS DEPENDENCIAS ---
l = 0.64
epsilon = 1445 + 35
el = l * epsilon

def model_4(t, c1, k, x1, y1):
    """
    Define el modelo 4 para la predicción de absorbancia.
    """
    return el * (x1 - y1 * np.real(lambertw(c1 * np.exp(-k * t))))

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Monitoreo de Nanopartículas en Tiempo Real",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TÍTULO Y DESCRIPCIÓN ---
st.title("🧪 Monitoreo de Formación de Nanopartículas en Tiempo Real")
st.markdown("""
Esta aplicación simula la formación de nanopartículas en tiempo real, basándose en un modelo matemático
previamente entrenado. La gráfica muestra la evolución de la absorbancia, un indicador de la concentración
de nanopartículas, mientras que la tabla registra los datos a intervalos específicos.
""")

# --- CARGAR EL MODELO ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "files", "models", "model_4.pkl")

@st.cache_resource
def load_model(path):
    """Carga el modelo y sus parámetros desde un archivo pickle."""
    try:
        with open(path, 'rb') as file:
            model_data = pickle.load(file)
        return model_data['model'], model_data['parameters']
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo del modelo en la ruta: {path}")
        st.stop()
    except Exception as e:
        st.error(f"Ocurrió un error al cargar el modelo: {e}")
        st.stop()

model, params = load_model(MODEL_PATH)

# --- FUNCIONES DE SIMULACIÓN ---
def model_4_prediction(t_values, c1, k, x1, y1):
    """Predice la absorbancia usando el modelo 4 con los parámetros dados."""
    return el * (x1 - y1 * np.real(lambertw(c1 * np.exp(-k * t_values))))

def simulate_data_point(current_time_seconds, model_func, model_params, noise_level=0.01):
    """Simula un único punto de datos con ruido gaussiano."""
    time_min = current_time_seconds / 60.0
    predicted_absorbance = model_func(np.array([time_min]), *model_params)[0]
    noise = np.random.normal(0, noise_level * predicted_absorbance)
    simulated_absorbance = predicted_absorbance + noise
    return time_min, simulated_absorbance

# --- INTERFAZ DE USUARIO Y LÓGICA DE ESTADO ---
st.sidebar.header("🕹️ Control de Simulación")

# Columnas para los botones para una mejor apariencia
col1, col2 = st.sidebar.columns(2)
with col1:
    start_simulation = st.button("▶️ Iniciar")
with col2:
    stop_simulation = st.button("⏹️ Detener")

# Inicialización del estado de la sesión
if 'running' not in st.session_state:
    st.session_state.running = False
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["Tiempo (min)", "Absorbancia (u.a.)"])
if 'current_time' not in st.session_state:
    st.session_state.current_time = 0
if 'last_table_update' not in st.session_state:
    st.session_state.last_table_update = -5

# Lógica de los botones
if start_simulation:
    st.session_state.running = True
    st.session_state.data = pd.DataFrame(columns=["Tiempo (min)", "Absorbancia (u.a.)"])
    st.session_state.current_time = 0
    st.session_state.last_table_update = -5
    st.rerun() # Forzar la actualización para reflejar el estado 'running'

if stop_simulation:
    st.session_state.running = False
    st.sidebar.success("Simulación detenida.")
    st.rerun()

# --- VISUALIZACIÓN DINÁMICA ---
st.header("📈 Gráfica de Absorbancia vs. Tiempo")
chart_placeholder = st.empty()

st.header("📋 Registros de Datos")
table_placeholder = st.empty()

# Mostrar estado inicial o final
if not st.session_state.running:
    st.info("Presiona 'Iniciar' para comenzar la simulación.")
    # Mostrar datos finales si existen
    if not st.session_state.data.empty:
        final_df = st.session_state.data.set_index("Tiempo (min)")
        chart_placeholder.line_chart(final_df)
        table_placeholder.dataframe(final_df.style.format({"Absorbancia (u.a.)": "{:.4f}"}))

# Bucle principal de simulación
if st.session_state.running:
    st.sidebar.info("Simulación en curso...")

    max_sim_time_seconds = 180 * 60  # 180 minutos en segundos

    while st.session_state.running and st.session_state.current_time <= max_sim_time_seconds:
        time_min, simulated_absorbance = simulate_data_point(
            st.session_state.current_time, model_4_prediction, params
        )

        new_row = pd.DataFrame([{
            "Tiempo (min)": time_min,
            "Absorbancia (u.a.)": simulated_absorbance
        }])
        st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)

        # --- Actualización de la UI ---
        # El gráfico se actualiza en cada paso para una visualización fluida
        chart_placeholder.line_chart(st.session_state.data.set_index("Tiempo (min)"))

        # La tabla se actualiza cada 5 segundos
        if (st.session_state.current_time - st.session_state.last_table_update) >= 5:
            # Mostramos los últimos 10 registros
            table_placeholder.dataframe(
                st.session_state.data.tail(10).set_index("Tiempo (min)").style.format({"Absorbancia (u.a.)": "{:.4f}"})
            )
            st.session_state.last_table_update = st.session_state.current_time

        st.session_state.current_time += 1
        time.sleep(0.05) # Pausa para una simulación más fluida

    if st.session_state.current_time > max_sim_time_seconds:
        st.session_state.running = False
        st.sidebar.success("Simulación completada.")
        st.rerun()