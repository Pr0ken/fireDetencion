from flask import Flask, render_template, request
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Cargar modelo
modelo = load_model('modelo.h5')  # Ajusta el nombre si tu modelo es diferente

# Función para predecir
def predecir_imagen(ruta_imagen):
    img = Image.open(ruta_imagen).resize((224, 224))  # Ajusta al tamaño que usa tu modelo
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediccion = modelo.predict(img_array)
    return "Incendio" if prediccion[0][0] > 0.5 else "No incendio"  # Ajusta si tienes más de una clase

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    imagen_url = None
    if request.method == 'POST':
        archivo = request.files['imagen']
        if archivo:
            ruta = os.path.join(app.config['UPLOAD_FOLDER'], archivo.filename)
            archivo.save(ruta)
            resultado = predecir_imagen(ruta)
            imagen_url = ruta
    return render_template('index.html', resultado=resultado, imagen=imagen_url)

if __name__ == '__main__':
    app.run(debug=True)
