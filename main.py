import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from lectura_csv import normalizarDatos

x_normalizado, y_normalizado = normalizarDatos()

# Convertir los datos normalizados a arrays de numpy
x = np.array(x_normalizado)
y = np.array(y_normalizado)

# Crear el modelo secuencial
modelo = Sequential()

# Añadir una capa al modelo para implementar la regresión lineal
modelo.add(Dense(1, input_shape=(1,), activation='linear'))

# Crear el optimizador SGD con una tasa de aprendizaje específica
sgd = SGD(learning_rate=0.0001)

# Compilar el modelo especificando la función de pérdida y el optimizador
modelo.compile(loss='mse', optimizer=sgd)

# Imprimir un resumen del modelo
modelo.summary()

# Entrenamiento del modelo
num_epochs = 10
batch_size = x.shape[0]
historia = modelo.fit(x, y, epochs=num_epochs, batch_size=batch_size, verbose=1)

# Obtener los parámetros w y b calculados
capas = modelo.layers[0]
w, b = capas.get_weights()
print('Parámetros: w = {:.1f}, b = {:.1f}'.format(w[0][0], b[0]))

# Graficar el comportamiento de la pérdida (ECM) durante el entrenamiento
plt.subplot(1, 2, 1)
plt.plot(historia.history['loss'])
plt.xlabel('Época')
plt.ylabel('ECM')
plt.title('ECM vs. Épocas')

# Graficar el resultado de la regresión superpuesto a los datos originales
y_regr = modelo.predict(x)
plt.subplot(1, 2, 2)
plt.scatter(x, y)
plt.plot(x, y_regr, 'r')
plt.title('Datos originales y regresión lineal')
plt.show()

