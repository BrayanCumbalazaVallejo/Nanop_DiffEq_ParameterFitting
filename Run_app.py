import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os
from scipy.special import lambertw

# Configuración de la página
st.set_page_config(
    page_title="Monitoreo de Concentración en Tiempo Real",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧪 Monitoreo de Concentración en Tiempo Real")

# --- Cargar el modelo ---
# Ruta relativa al directorio 'models' dentro de 'files'
MODEL_PATH = os.path.join(os.path.dirname(__file__), "files", "models", "model_4.pkl")

@st.cache_resource
def load_model(path):
    """Carga el modelo y sus parámetros."""
    try:
        with open(path, 'rb') as file:
            model_data = pickle.load(file)
        return model_data['model'], model_data['parameters']
    except FileNotFoundError:
        st.error(f"Error: El archivo del modelo no se encuentra en la ruta: {path}")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.stop()

model, params = load_model(MODEL_PATH)

# Definir longitud de la celda y absortividad molar (como en tu notebook)
l = 0.64  # Longitud de la celda en cm
epsilon = 1445 + 35  # Absortividad molar
el = l * epsilon  # Absortividad molar por cm

# Funciones de utilidad para simulación
def model_4_prediction(t_values, c1, k, x1, y1):
    """
    Predice la absorbancia usando el modelo 4 con los parámetros dados.
    Esta función es una réplica del modelo en tu notebook, asegurando que 'el' se use.
    """
    return el * (x1 - y1 * np.real(lambertw(c1 * np.exp(-k * t_values))))

def simulate_data(current_time_point, total_data_points, model_func, model_params, noise_level=0.01):
    """
    Simula un punto de datos basado en el modelo y añade ruido.
    """
    if current_time_point >= total_data_points:
        return None
    
    time_min = current_time_point * 5 / 60  # Asumiendo registros cada 5 segundos
    predicted_absorbance = model_func(np.array([time_min]), *model_params)[0]
    
    # Añadir ruido gaussiano
    noise = np.random.normal(0, noise_level * predicted_absorbance)
    simulated_absorbance = predicted_absorbance + noise
    
    return time_min, simulated_absorbance

# --- Interfaz de usuario de Streamlit ---

st.sidebar.header("Control de Simulación")
start_simulation = st.sidebar.button("▶️ Iniciar Simulación")
stop_simulation = st.sidebar.button("⏹️ Detener Simulación")

# Inicializar estado de la simulación
if 'running' not in st.session_state:
    st.session_state.running = False
if 'data_frame' not in st.session_state:
    st.session_state.data_frame = pd.DataFrame(columns=["Tiempo (min)", "Absorbancia (u.a.)"])
if 'current_point_index' not in st.session_state:
    st.session_state.current_point_index = 0

if start_simulation:
    st.session_state.running = True
    st.session_state.data_frame = pd.DataFrame(columns=["Tiempo (min)", "Absorbancia (u.a.)"])
    st.session_state.current_point_index = 0

if stop_simulation:
    st.session_state.running = False
    st.sidebar.success("Simulación detenida.")

# Contenedores para actualizar la gráfica y la tabla
chart_placeholder = st.empty()
table_placeholder = st.empty()

if st.session_state.running:
    st.sidebar.info("Simulación en curso...")
    max_time_in_notebook = 180  # Tiempo máximo de simulación en tu gráfica del notebook
    total_simulated_points = int(max_time_in_notebook * 60 / 5) # Calcular total de puntos para 5 seg de intervalo

    while st.session_state.running and st.session_state.current_point_index <= total_simulated_points:
        time_min, simulated_absorbance = simulate_data(
            st.session_state.current_point_index,
            total_simulated_points,
            model_4_prediction,
            params
        )

        if time_min is None: # Si ya no hay más puntos para simular
            st.session_state.running = False
            st.sidebar.warning("Simulación completada. No hay más datos.")
            break

        # Añadir el nuevo punto al DataFrame
        new_row = pd.DataFrame([{
            "Tiempo (min)": time_min,
            "Absorbancia (u.a.)": simulated_absorbance
        }])
        st.session_state.data_frame = pd.concat([st.session_state.data_frame, new_row], ignore_index=True)

        # Actualizar gráfica
        with chart_placeholder:
            st.line_chart(st.session_state.data_frame.set_index("Tiempo (min)"))

        # Actualizar tabla (mostrar solo los últimos 10 registros para evitar sobrecarga)
        with table_placeholder:
            st.subheader("Últimos Registros")
            st.dataframe(st.session_state.data_frame.tail(10).style.format({"Absorbancia (u.a.)": "{:.4f}"}))

        st.session_state.current_point_index += 1
        time.sleep(0.5)  # Esperar 0.5 segundos para simular tiempo real y no saturar la CPU
        
    if st.session_state.current_point_index > total_simulated_points:
        st.session_state.running = False
        st.sidebar.success("Simulación terminada.")

else:
    st.info("Presiona 'Iniciar Simulación' para comenzar a ver los datos en tiempo real.")
    if not st.session_state.data_frame.empty:
        chart_placeholder.line_chart(st.session_state.data_frame.set_index("Tiempo (min)"))
        table_placeholder.subheader("Últimos Registros")
        table_placeholder.dataframe(st.session_state.data_frame.tail(10).style.format({"Absorbancia (u.a.)": "{:.4f}"}))