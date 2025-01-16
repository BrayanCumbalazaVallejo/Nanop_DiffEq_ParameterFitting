import numpy as np
import matplotlib.pyplot as plt
# Vectores organizados
tiempo = np.array([2.50, 7.50, 12.50, 17.50, 22.50, 27.50, 32.50, 37.50, 52.50, 62.50, 
                   72.50, 82.50, 92.50, 122.50, 132.50, 142.50, 152.50, 162.50, 172.50, 182.50])

absorbancia = np.array([0.1729, 0.3468, 0.5109, 0.6651, 0.8095, 0.9441, 1.0688, 1.1837, 1.2887, 
                        1.3840, 1.4694, 1.5449, 1.6107, 1.6666, 1.7127, 1.7489, 1.7753, 1.7919, 
                        1.7986, 1.7956])

# Gráfica
plt.figure(figsize=(10, 6))
plt.plot(tiempo, absorbancia, 'o-', color='b', label='Promedio Suavizado')
plt.title('Promedio Suavizado de Absorbancia en el Tiempo')
plt.xlabel('Tiempo (min)')
plt.ylabel('Absorbancia (u.a.)')
plt.legend()
plt.grid(True)
plt.show()
