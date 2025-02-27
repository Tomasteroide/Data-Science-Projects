# Descenso de gradiente estocástico

Para que el algoritmo de aprendizaje funcione más rápido, podemos reducir el tiempo de una iteración.
Podemos calcular el gradiente usando pequeñas partes del conjunto de entrenamiento. Estas piezas se conocen como minilotes o lotes. Para que el algoritmo "vea" todo el conjunto de entrenamiento, deben cambiarse sus lotes en cada iteración de forma aleatoria. Aquí necesitamos el descenso de gradiente estocástico en minilotes o descenso de gradiente estocástico, DGE (del griego στοχαστικός, "capaz de adivinar"). Ayuda a acelerar el entrenamiento del modelo.

Para obtener lotes, necesitamos barajar todos los datos del conjunto de entrenamiento y dividirlo en partes. Un lote debe contener un promedio de 128 observaciones (u otro valor potencia de 2, e.g., 64, 256, 512, etc). Cuando el algoritmo DGE ha pasado por todos los lotes una vez, significa que una época ha terminado. El número de épocas depende del tamaño del conjunto de entrenamiento. Pueden ser una o dos si el conjunto es pequeño o varias docenas si el conjunto es grande. El número de lotes es igual al número de iteraciones para completar una época.

Así es como funciona el algoritmo DGE:

1. Hiperparámetros de entrada: tamaño del lote, número de épocas y tamaño del paso.
2. Define los valores iniciales de los pesos del modelo.
3. Divide el conjunto de entrenamiento en lotes para cada época.
4. Para cada lote: 
 1. Calcula el gradiente de la función de pérdida;
 2. Actualiza los pesos del modelo (agrega el gradiente negativo multiplicado por el tamaño del paso a los pesos actuales).
5. El algoritmo devuelve los pesos finales del modelo.


# Neural Networks

