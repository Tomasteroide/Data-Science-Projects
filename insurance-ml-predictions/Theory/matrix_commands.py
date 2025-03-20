# Matrix creation

import numpy as np
matrix = np.array([[3, 14, 159],[-2, 7, 183]])
print(matrix)

# DF to matrix
import pandas as pd

monday_df = pd.DataFrame({'Minutos': [10, 3, 15, 27, 7], 
                          'Mensajes': [2, 5, 3, 0, 1], 
                          'Megabytes': [72, 111, 50, 76, 85]})
 
monday = monday_df.values #.values creates de matrix

print(monday)
print(monday[2]) # third row
print(monday[:,2]) # third column

# Matrix operations

# Multiplicación de matrices -> tres formas equivalentes de multiplicar

A = np.array([
    [1, 2, 3], 
    [-1, -2, -3]])

B = np.array([
    [1, 0], 
    [0, 1],
    [1, 1]])

print(np.dot(A,B)) 
print(A.dot(B)) 
print(A @ B)

