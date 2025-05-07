from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Cargar modelo
modelo = load_model('fire_detection_model.keras')  

# Función para predecir
def predecir_imagen(ruta_imagen):
    img = Image.open(ruta_imagen).resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediccion = modelo.predict(img_array)
    prob_no_fire = prediccion[0][0]
    print(f"Probabilidad no incendio: {prob_no_fire:.4f}")
    if prob_no_fire > 0.5:
        return f"No incendio ({prob_no_fire:.2%})"
    else:
        return f"Incendio ({1 - prob_no_fire:.2%})"
    

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
