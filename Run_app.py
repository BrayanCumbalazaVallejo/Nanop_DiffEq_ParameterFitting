import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os
from scipy.special import lambertw

# --- DEFINICIÓN DE LA FUNCIÓN MODEL_4 Y SUS DEPENDENCIAS ---
# Estas variables y la función deben estar definidas en Run_app.py
# para que pickle pueda encontrarlas al cargar el modelo.
l = 0.64  # Longitud de la celda en cm
epsilon = 1445 + 35  # Absortividad molar
el = l * epsilon  # Absortividad molar por cm

def model_4(t, c1, k, x1, y1):
    """
    Define el modelo 4 para la predicción de absorbancia.
    """
    return el * (x1 - y1 * np.real(lambertw(c1 * np.exp(-k * t))))

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Monitoreo de Concentración en Tiempo Real",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧪 Monitoreo de Concentración en Tiempo Real")

# --- Cargar el modelo ---
# La ruta se construye de forma robusta utilizando os.path.dirname(__file__)
# Esto asume que 'Run_app.py' está en el mismo directorio que la carpeta 'files'.
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

# Cargar el modelo y sus parámetros
model, params = load_model(MODEL_PATH)

# Funciones de utilidad para simulación
def model_4_prediction(t_values, c1, k, x1, y1):
    """
    Predice la absorbancia usando el modelo 4 con los parámetros dados.
    """
    # Asegúrate de que 'el' esté definido globalmente o pasado como argumento si es necesario.
    # En este caso, ya está definido globalmente arriba.
    return el * (x1 - y1 * np.real(lambertw(c1 * np.exp(-k * t_values))))

def simulate_data_point(current_time_seconds, model_func, model_params, noise_level=0.01):
    """
    Simula un punto de datos basado en el modelo para un tiempo dado en segundos.
    """
    time_min = current_time_seconds / 60.0
    predicted_absorbance = model_func(np.array([time_min]), *model_params)[0]
    
    # Añadir ruido gaussiano
    noise = np.random.normal(0, noise_level * predicted_absorbance)
    simulated_absorbance = predicted_absorbance + noise
    
    return time_min, simulated_absorbance

# --- Interfaz de usuario de Streamlit ---

st.sidebar.header("Control de Simulación")
start_simulation = st.sidebar.button("▶️ Iniciar Simulación")
stop_simulation = st.sidebar.button("⏹️ Detener Simulación")

# Inicializar estado de la simulación usando st.session_state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'data_frame_all' not in st.session_state: # DataFrame para todos los puntos de la gráfica
    st.session_state.data_frame_all = pd.DataFrame(columns=["Tiempo (min)", "Absorbancia (u.a.)"])
if 'data_frame_table' not in st.session_state: # DataFrame para la tabla (cada 5 segundos)
    st.session_state.data_frame_table = pd.DataFrame(columns=["Tiempo (min)", "Absorbancia (u.a.)"])
if 'current_sim_time_seconds' not in st.session_state: # Tiempo actual de simulación en segundos
    st.session_state.current_sim_time_seconds = 0
if 'last_table_update_time' not in st.session_state: # Último tiempo en que se actualizó la tabla
    st.session_state.last_table_update_time = -5 # Inicializar para que la primera actualización ocurra en t=0

# Lógica para iniciar la simulación
if start_simulation:
    st.session_state.running = True
    st.session_state.data_frame_all = pd.DataFrame(columns=["Tiempo (min)", "Absorbancia (u.a.)"])
    st.session_state.data_frame_table = pd.DataFrame(columns=["Tiempo (min)", "Absorbancia (u.a.)"])
    st.session_state.current_sim_time_seconds = 0
    st.session_state.last_table_update_time = -5 # Resetea para que la tabla se actualice en t=0

# Lógica para detener la simulación
if stop_simulation:
    st.session_state.running = False
    st.sidebar.success("Simulación detenida.")

# Contenedores para actualizar la gráfica y la tabla dinámicamente
chart_placeholder = st.empty()
table_placeholder = st.empty()

# Bucle de simulación en tiempo real
if st.session_state.running:
    st.sidebar.info("Simulación en curso...")
    
    # Tiempo máximo de simulación en minutos (según tu gráfica de model_4_comparison.png)
    max_time_in_notebook_mins = 180 
    max_sim_time_seconds = max_time_in_notebook_mins * 60 # Convertir a segundos

    while st.session_state.running and st.session_state.current_sim_time_seconds <= max_sim_time_seconds:
        # Simular un nuevo punto de datos cada segundo (o el intervalo deseado)
        time_min, simulated_absorbance = simulate_data_point(
            st.session_state.current_sim_time_seconds,
            model_4_prediction,
            params
        )

        # Añadir el nuevo punto al DataFrame para la gráfica
        new_row_all = pd.DataFrame([{
            "Tiempo (min)": time_min,
            "Absorbancia (u.a.)": simulated_absorbance
        }])
        st.session_state.data_frame_all = pd.concat([st.session_state.data_frame_all, new_row_all], ignore_index=True)

        # Actualizar gráfica en el placeholder con todos los puntos
        with chart_placeholder:
            st.line_chart(st.session_state.data_frame_all.set_index("Tiempo (min)"))

        # Actualizar tabla de registros cada 5 segundos
        if (st.session_state.current_sim_time_seconds - st.session_state.last_table_update_time) >= 5:
            new_row_table = pd.DataFrame([{
                "Tiempo (min)": time_min,
                "Absorbancia (u.a.)": simulated_absorbance
            }])
            st.session_state.data_frame_table = pd.concat([st.session_state.data_frame_table, new_row_table], ignore_index=True)
            
            with table_placeholder:
                st.subheader("Últimos Registros")
                st.dataframe(st.session_state.data_frame_table.tail(10).style.format({"Absorbancia (u.a.)": "{:.4f}"}))
            
            st.session_state.last_table_update_time = st.session_state.current_sim_time_seconds

        st.session_state.current_sim_time_seconds += 1 # Avanzar un segundo
        time.sleep(0.1)  # Pausa corta para permitir la actualización de la UI. Ajusta este valor si es necesario.
                         # Un valor menor hará que la simulación sea más rápida, pero podría afectar el renderizado de la UI.

    # Mensaje final si la simulación se completa (sin detenerla manualmente)
    if st.session_state.current_sim_time_seconds > max_sim_time_seconds:
        st.session_state.running = False
        st.sidebar.success("Simulación terminada.")

# Estado inicial o cuando la simulación no está corriendo
else:
    st.info("Presiona 'Iniciar Simulación' para comenzar a ver los datos en tiempo real.")
    # Si ya hay datos en la sesión (por una simulación anterior o al recargar), mostrarlos
    if not st.session_state.data_frame_all.empty:
        chart_placeholder.line_chart(st.session_state.data_frame_all.set_index("Tiempo (min)"))
    if not st.session_state.data_frame_table.empty:
        table_placeholder.subheader("Últimos Registros")
        table_placeholder.dataframe(st.session_state.data_frame_table.tail(10).style.format({"Absorbancia (u.a.)": "{:.4f}"}))