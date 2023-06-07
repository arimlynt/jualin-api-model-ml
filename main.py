import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from flask import Flask, jsonify, request
import os
from google.cloud import storage
import io

# Define Class Names
class_names = ['Ayam Goreng Tepung', 'Baju', 'Bakso', 'Celana', 'Dompet',
               'Gaun', 'Jam Tangan', 'Jus', 'Kacamata', 'Kaos Kaki',
               'Keripik', 'Kopi', 'MakeUp', 'Martabak Manis', 'Mie Ayam',
               'Mie Goreng', 'Nasi Goreng', 'Parfum', 'Pisang Goreng', 'Roti Bakar',
               'Sate Ayam', 'Sendal', 'Sepatu', 'Tas', 'Topi']

app = Flask(__name__)

# Load The Model
model = load_model('model/model.h5')

# Configure Google Cloud Storage
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account-key.json"
storage_client = storage.Client()
bucket_name = 'api-model-ml'  # Ganti dengan nama bucket Anda

@app.route('/predict', methods=['POST'])
def predictions():
    file = request.files['imagefile']
    
    try:
        img = image.load_img(io.BytesIO(file.stream.read()), target_size=(224,224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        img = np.vstack([img])

        pred = model.predict(img)
        classes = int(pred.argmax(axis=-1))
        result = class_names[classes]

        # Upload image to Google Cloud Storage
        bucket = storage_client.bucket(bucket_name)

        # Hapus foto lama jika ada
        if bucket.blob(file.filename).exists():
            bucket.blob(file.filename).delete()

        blob = bucket.blob(file.filename)
        file_data = file.read()
        blob.upload_from_string(file_data, content_type=file.content_type)

        return jsonify(str(result))
    except Exception as e:
        print(e)
        return jsonify({'Error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)