import pandas as pd
import requests
import joblib
import json
import numpy as np

# URL del script PHP
url_php_script = 'https://stressappshirleybd.000webhostapp.com/model.php'

# Realizar solicitud al script PHP
response = requests.get(url_php_script)
data = response.json()

# Crear un DataFrame con los resultados
df_combined = pd.DataFrame(data, index=[0])
print(df_combined)

# Cargar el modelo previamente entrenado
modelo = joblib.load('Random Forest_model.pkl')

# Realizar predicciones en los datos combinados
predicciones = modelo.predict(df_combined.values.reshape(1, -1))

# Imprimir las predicciones
print("Predicciones:")
print(predicciones)

if np.any(predicciones == 0):
    print("Se detecto un estres bajo")
elif np.any(predicciones == 1):
    print("Se detecto un estres moderado")
else:
    print("Se detecto un estres alto")
