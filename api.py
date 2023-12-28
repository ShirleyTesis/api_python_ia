from flask import Flask, jsonify
import requests
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET'])
def predict_stress():
    # Tu código para obtener datos y realizar predicciones
    # (Simulando el comportamiento actual con datos estáticos)
    url_php_script = 'https://stressappshirleybd.000webhostapp.com/model.php'
    response = requests.get(url_php_script)
    data = response.json()
    df_combined = pd.DataFrame(data, index=[0])
    
    modelo = joblib.load('Random Forest_model.pkl')
    predicciones = modelo.predict(df_combined.values.reshape(1, -1))

    # Mapeo de predicciones a respuestas
    if np.any(predicciones == 0):
        resultado = "Estres bajo"
    elif np.any(predicciones == 1):
        resultado = "Estres moderado"
    else:
        resultado = "Estres alto"

    # Devolver el resultado como JSON
    return jsonify({"resultado": resultado})

if __name__ == '__main__':
    app.run(debug=True)
