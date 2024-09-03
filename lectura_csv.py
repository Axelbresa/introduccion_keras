import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import csv
import pandas as pd
import matplotlib.pyplot as plt

altura=[]
peso=[]

# Paso 1: Leer los datos
with open("altura_peso.csv")as read_csv:
    file_csv=csv.reader(read_csv)
    for columna in file_csv:
        if columna[0].isdigit() and columna[1].isdigit():
            altura.append(int(columna[0]))  
            peso.append(int(columna[1]))  


# Paso 2: Crear un DataFrame con los datos
data = pd.DataFrame({
    'x': altura,
    'y': peso
})

# Mostrar el DataFrame

def normalizarDatos():
    x = (data["x"] - np.mean(data["x"])) / np.std(data["x"])
    y = (data["y"] - np.mean(data["y"])) / np.std(data["y"])
    return x, y

x_normalizado, y_normalizado = normalizarDatos()