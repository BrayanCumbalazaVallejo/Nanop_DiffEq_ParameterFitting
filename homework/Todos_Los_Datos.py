import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Datos experimentales originales
t_exp = np.array([0, 5, 10, 15, 20, 25, 30, 35, 50, 60, 70, 80, 90, 120, 130, 140, 150, 160, 170, 180,
                  0, 5, 10, 15, 20, 25, 30, 35, 50, 60, 70, 80, 90, 120, 130, 140, 150, 160, 170, 180,
                  0, 5, 10, 15, 20, 25, 30, 35, 50, 60, 70, 80, 90, 120, 130, 140, 150, 160, 170, 180]) + 2.5

A_exp = np.array([1.026, 1.417, 1.784, 2.648, 1.531, 2.14, 2.247, 2.189, 1.973, 1.87, 1.694, 2.01, 2.832, 2.486, 2.378, 3.303, 3.101, 2.939, 3.058, 2.9, 
                  1.146, 1.343, 1.34, 1.554, 1.671, 3.136, 2.626, 2.731, 2.532, 2.647, 2.299, 2.902, 2.505, 3.273, 3.008, 2.819, 3.344, 2.156, 3.188, 2.176, 
                  1.123, 1.187, 1.516, 1.352, 1.407, 1.933, 2.169, 2.298, 1.751, 2.234, 1.936, 3.06, 2.362, 2.646, 2.332, 2.979, 2.918, 2.749, 2.759, 2.794]) - 1.026

# Configuración para el suavizado
n_corridas = 3  # Número de corridas
n_puntos_por_corrida = len(A_exp) // n_corridas
window_length = 20  # Longitud de la ventana (debe ser impar)
polyorder = 2  # Orden del polinomio

# Reestructuramos los datos en corridas
A_exp_corridas = A_exp.reshape(n_corridas, n_puntos_por_corrida)
t_exp_corridas = t_exp.reshape(n_corridas, n_puntos_por_corrida)

# Aplicamos el filtro de Savitzky-Golay
A_suavizado_corridas = np.array([
    savgol_filter(A_exp_corrida, window_length, polyorder) for A_exp_corrida in A_exp_corridas
])

# Calcular el promedio de las corridas suavizadas
t_promedio = t_exp_corridas.mean(axis=0)  # El tiempo promedio entre las corridas (idéntico para todas)
A_promedio_suavizado = A_suavizado_corridas.mean(axis=0)  # Promedio de las corridas suavizadas

# Graficar datos originales, suavizados por corrida y promedio
plt.figure(figsize=(12, 8))

# Gráfica de las corridas suavizadas y originales
for i in range(n_corridas):
    plt.plot(t_exp_corridas[i], A_exp_corridas[i], 'o', label=f'Corrida {i + 1} (Original)', alpha=0.5)
    plt.plot(t_exp_corridas[i], A_suavizado_corridas[i], '-', label=f'Corrida {i + 1} (Suavizado)')

# Gráfica del promedio suavizado
plt.plot(t_promedio, A_promedio_suavizado, 'k--', linewidth=2, label='Promedio Suavizado')

# Configuración de la gráfica
plt.title('Promedio Suavizado de Datos de Absorbancia con Savitzky-Golay')
plt.xlabel('Tiempo (min)')
plt.ylabel('Absorbancia (u.a.)')
plt.legend()
plt.grid(True)
plt.show()

# Imprimir el promedio suavizado
print("Datos promedio suavizados:")
for t, a in zip(t_promedio, A_promedio_suavizado):
    print(f"Tiempo: {t:.2f} min, Absorbancia promedio: {a:.4f}")
