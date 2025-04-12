import numpy as np


def convolve(sequence, weights):
    convolution = np.zeros(len(sequence) - len(weights) + 1)
    for i in range(convolution.shape[0]):
        convolution[i] = np.sum(
            np.array(weights) * np.array(sequence[i : i + len(weights)])
        )
    return convolution

def convolve_2d(sequence, weights):

    sequence = np.array(sequence)
    weights = np.array(weights)

    if sequence.ndim != 2:
        raise ValueError("sequence debe ser un array 2D")
    if weights.ndim != 2:
        raise ValueError("weights debe ser un array 2D")
    
    seq_height, seq_width = sequence.shape
    w_height, w_width = weights.shape
    

    output_height = seq_height - w_height + 1
    output_width = seq_width - w_width + 1
    

    convolution = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):

            subarray = sequence[i:i + w_height, j:j + w_width]
            convolution[i, j] = np.sum(subarray * weights)
    
    return convolution

def convolve(sequence, weights):
    # Aseguramos que sequence y weights sean numpy arrays
    sequence = np.array(sequence)
    weights = np.array(weights)
    
    # Verificamos que sequence y weights sean matrices 3D
    if sequence.ndim != 3:
        raise ValueError("sequence debe ser un array 3D")
    if weights.ndim != 3:
        raise ValueError("weights debe ser un array 3D")
    
    # Dimensiones de la secuencia y los pesos
    seq_depth, seq_height, seq_width = sequence.shape
    w_depth, w_height, w_width = weights.shape
    
    # Verificamos que las dimensiones del filtro no sean mayores que las de la imagen
    if seq_depth < w_depth or seq_height < w_height or seq_width < w_width:
        raise ValueError("El tamaño del filtro no debe ser mayor que el tamaño de la imagen en ninguna dimensión.")
    
    # Calculamos el tamaño del resultado de la convolución
    output_depth = seq_depth - w_depth + 1
    output_height = seq_height - w_height + 1
    output_width = seq_width - w_width + 1
    
    # Inicializamos el array de convolución con ceros
    convolution = np.zeros((output_depth, output_height, output_width))
    
    # Realizamos la convolución 3D
    for i in range(output_depth):
        for j in range(output_height):
            for k in range(output_width):
                # Tomamos un subarray de la secuencia y realizamos la convolución
                subarray = sequence[i:i + w_depth, j:j + w_height, k:k + w_width]
                convolution[i, j, k] = np.sum(subarray * weights)
    
    return convolution


s = [[2, 1, 3], 
     [4, 0, 2], 
     [1, 5, 6]]

w = [[-1, -2], 
     [1,   2]]

print(convolve_2d(s, w))