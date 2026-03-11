
from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from datetime import datetime
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

model = None
MODEL_ERROR = None
MODEL_PATH = None
CLASS_NAMES = ['Parasitized', 'Uninfected']

print("\n" + "="*70)
print("MALARIA DETECTION - LOADING MODEL")
print("="*70)

# ---------------- LOAD MODEL ---------------- #

try:
    print("TensorFlow imported successfully")
except Exception as e:
    MODEL_ERROR = f"Import failed: {e}"
    print(f"❌ {MODEL_ERROR}")

if MODEL_ERROR is None:
    current_dir = os.getcwd()
    print(f"Directory: {current_dir}")

    try:
        h5_files = [f for f in os.listdir(current_dir) if f.endswith('.h5')]

        if h5_files:
            MODEL_PATH = os.path.join(current_dir, h5_files[0])
            print(f"Found model file: {h5_files[0]}")

            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

            print(f"Model Loaded | Input: {model.input_shape} Output: {model.output_shape}")

        else:
            MODEL_ERROR = "No .h5 file found"
            print(f"❌ {MODEL_ERROR}")

    except Exception as e:
        MODEL_ERROR = str(e)
        print(f"❌ {MODEL_ERROR}")


# ---------------- IMAGE PREPROCESSING ---------------- #

def prepare_image(image_file):

    try:

        img = Image.open(image_file)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        input_shape = model.input_shape
        img_size = (input_shape[1], input_shape[2])

        img = img.resize(img_size)

        img_array = np.array(img, dtype='float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        print(f"Image Processing Error: {e}")
        return None



# ---------------- HOME PAGE ---------------- #

@app.route('/')
def home():
    return render_template("index.html")

# ---------------- UPLOAD PAGE ---------------- #

@app.route('/upload')
def upload():
    return render_template("upload.html")




# ---------------- PREDICTION API ---------------- #

@app.route('/predict', methods=['POST'])
def predict():

    if model is None:
        return render_template(
            "report.html",
            result="Model Not Loaded",
            confidence="0%"
        )

    try:

        if 'file' not in request.files:
            return render_template(
                "report.html",
                result="No Image Uploaded",
                confidence="0%"
            )

        file = request.files['file']

        if file.filename == '':
            return render_template(
                "report.html",
                result="No File Selected",
                confidence="0%"
            )

        img_array = prepare_image(file)

        if img_array is None:
            return render_template(
                "report.html",
                result="Image Processing Failed",
                confidence="0%"
            )

        prediction = model.predict(img_array)
        prob = float(prediction[0][0])

        if prob > 0.5:
            result = "Uninfected"
            confidence = prob * 100
        else:
            result = "Parasitized"
            confidence = (1 - prob) * 100

        return render_template(
            "report.html",
            result=result,
            confidence=f"{confidence:.2f}%"
        )

    except Exception as e:

        print(f"Prediction Error: {e}")

        return render_template(
            "report.html",
            result="Prediction Error",
            confidence="0%"
        )

# ---------------- HEALTH CHECK ---------------- #

@app.route('/health')
def health():

    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# ---------------- RUN SERVER ---------------- #

if __name__ == "__main__":

    print("\n" + "="*60)

    if model:
        print("Model Loaded Successfully")
    else:
        print("Model NOT Loaded")

    print("Server Running at:")
    print("http://127.0.0.1:5000")

    print("="*60 + "\n")


    app.run(host="0.0.0.0", port=10000)
