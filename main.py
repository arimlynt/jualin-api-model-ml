import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from flask import Flask, jsonify, request

# Define Class Names
class_names = ['Ayam Goreng Tepung', 'Baju', 'Bakso', 'Celana', 'Dompet',
               'Gaun', 'Jam Tangan', 'Jus', 'Kacamata', 'Kaos Kaki',
               'Keripik', 'Kopi', 'MakeUp', 'Martabak Manis', 'Mie Ayam',
               'Mie Goreng', 'Nasi Goreng', 'Parfum', 'Pisang Goreng', 'Roti Bakar',
               'Sate Ayam', 'Sendal', 'Sepatu', 'Tas', 'Topi']

app = Flask(__name__)

# Load The Model
model = load_model('model/model.h5')

@app.route('/predict', methods=['POST'])
def predictions():
    file = request.files['imagefile']
    file_path = "./images/" + file.filename
    file.save(file_path)
    
    try:
        img = image.load_img(file_path, target_size=(224,224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        img = np.vstack([img])

        pred = model.predict(img)
        classes = int(pred.argmax(axis=-1))
        result = class_names[classes]

        os.remove(file_path)

        return jsonify(str(result))
    except Exception as e:
        print(e)
        return jsonify({'Error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
