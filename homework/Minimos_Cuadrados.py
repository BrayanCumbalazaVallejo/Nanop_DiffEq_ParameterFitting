#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Métodos Numéricos - Análisis numérico 
Universidad Nacional de Colombia, sede Medellín
Marzo 2025 

@Autores: Mauricio Osorio

Mínimos cuadrados

Calcula el polinomio que mejor se ajusta a un conjunto de puntos, de acuerdo con la
teoría de los mínimos cuadrados


Parámetros:
    x (array-like de tamaño (n, )): vector de abscisas.
    y (array-like de tamaño (n,)): vector de ordenadas.
    grado (int): grado del polinomio de ajuste 

    

Retorna:
    coeficientes (array-like de tamaño (n, )): vector con los coeficientes del polinomio ajustado
       en orden descendete de potencias de x

"""

import numpy as np
import matplotlib.pyplot as plt

def Minimos_Cuadrados(x, y, grado):

    n = len(x)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x = x.reshape(1, n)
    y = y.reshape(1, n)
    
    x = x.copy(); y = y.copy()
    
    
    
    if grado == 1:
        xmean = np.mean(x)
        ymean = np.mean(y)
        sumx2 = np.dot(x-xmean,np.transpose(x-xmean))
        sumxy = np.dot(y-ymean,np.transpose(x-xmean))
        
        A1 = sumxy/sumx2
        B1 = ymean - np.dot(A1,xmean)
        coeficientes = [A1,B1]
    else:
            
        # Construcción de la matriz del sistema normal X^T * X y del vector X^T * y
        F = np.zeros((n, grado + 1))
        B = np.zeros((1,grado + 1))

        for k in range(0, grado+1):
            F[:,k] = x**k
     
    
        A = np.dot(np.transpose(F),F)
        B = np.dot(np.transpose(F),np.transpose(y))
        coeficientes = np.linalg.solve(A, B)
        coeficientes = coeficientes.reshape(( grado +1,))
        coeficientes  = coeficientes[::-1]
        
              
    
    return [x.item() for x in coeficientes]

# # Datos de prueba
# x = [1, 2, 3, 4, 5, 6]
# # f = lambda x: x**2 + 2*x +1
# y = [ 4.2,  9.1, 15.8, 25.3, 35.5, 49]

# # Ajuste de un polinomio de grado 2
# grado = 2
# coeficientes = Minimos_Cuadrados(x, y, grado)

# # Generar puntos para graficar la curva ajustada
# x_fit = np.linspace(min(x), max(x), 100)
# y_fit = sum(coeficientes[i] * x_fit**i for i in range(grado + 1))

# # Mostrar resultados
# print(f"Coeficientes del polinomio ajustado (grado {grado}): {coeficientes}")

# # Graficar
# plt.scatter(x, y, color='blue', label='Datos')
# plt.plot(x_fit, y_fit, color='red', label=f'Ajuste de grado {grado}')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()

# otra manera de graficar es usando polyval de numpy.  PERO HAY QUE TENER CUIDADO! En numpy 
# los polinomios tienen orden de menor a mayor

# p = coeficientes[::-1]
# x_fit = np.linspace(min(x), max(x), 100)
# y_fit = np.polyval(p, x_fit)

# plt.scatter(x, y, color='blue', label='Datos')
# plt.plot(x_fit, y_fit, color='red', label=f'Ajuste de grado {grado}')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()



# --- Resolución del problema específico ---

# 1. Definir la función y los puntos de muestreo según la imagen
f = lambda x: 6 / (2 + 2 * x**2)
xL = np.linspace(-5, 4, 5)
yL = f(xL)

# 2. Aplicar el método de mínimos cuadrados para un ajuste de grado 1
grado = 1
coeficientes = Minimos_Cuadrados(xL, yL, grado)

print(f"\nCoeficientes del polinomio de grado {grado} (y = mx + b): {coeficientes}")
