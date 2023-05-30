from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Memuat model
model = load_model('model/model.h5')

# Fungsi untuk melakukan prediksi
def predict(image):
    # Mengubah ukuran gambar menjadi 224x224 piksel
    image = image.resize((224, 224))
    # Mengubah gambar menjadi array numpy
    image_array = np.asarray(image)
    # Menormalisasi array gambar
    normalized_image_array = (image_array.astype(np.float32) / 255.0)
    # Menambahkan dimensi tambahan agar sesuai dengan model
    input_data = np.expand_dims(normalized_image_array, axis=0)
    # Melakukan prediksi menggunakan model
    predictions = model.predict(input_data)
    # Mengambil label prediksi
    label = np.argmax(predictions[0])
    return str(label)  # Mengubah menjadi tipe data str

# Endpoint untuk melakukan prediksi
@app.route('/predict', methods=['POST'])
def run_prediction():
    # Menerima gambar dari permintaan POST
    image = request.files['image']
    # Membuka gambar menggunakan PIL
    img= Image.open(image)
    # Melakukan prediksi
    prediction = predict(img)
    # Menyusun respons dalam format JSON
    response = {'prediction': prediction}
    return jsonify(response)

if __name__ == '__main__':
    app.run()