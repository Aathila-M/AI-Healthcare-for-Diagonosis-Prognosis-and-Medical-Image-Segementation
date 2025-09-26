from flask import Flask, render_template, request, flash, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2

app = Flask(__name__)
app.secret_key = "your_secret_key"

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

MODEL_PATH = './models/model.h5'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
model = load_model(MODEL_PATH)

class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_tumor(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    index = np.argmax(prediction[0])
    confidence = prediction[0][index]
    label = class_labels[index]
    return ("No Tumor" if label == "notumor" else f"Tumor: {label.capitalize()}", confidence)

def add_overlay(image_path, result):
    img = cv2.imread(image_path)
    overlay = img.copy()
    color = (0, 255, 0) if result == "No Tumor" else (0, 0, 255)
    alpha = 0.3
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    new_path = image_path.replace(".", "_processed.")
    cv2.imwrite(new_path, img)
    return new_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            flash("No file selected")
            return redirect(request.url)

        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            result, confidence = predict_tumor(file_path)
            show_confidence = result != "No Tumor"
            processed_image = add_overlay(file_path, result)

            return render_template('index.html',
                                   result=result,
                                   confidence=f"{confidence*100:.2f}%" if show_confidence else "",
                                   show_confidence=show_confidence,
                                   file_path=processed_image)
        else:
            flash("Invalid file type. Only PNG, JPG, JPEG allowed.")
            return redirect(request.url)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
