from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from PIL import Image
import io
import cv2
import face_recognition
import os

DB_DIR = os.path.join(os.path.dirname(__file__), "../../face_db")

class ImageRecognitionModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def preprocess_image(self, img_path):
        img = load_img(img_path, target_size=(224, 224))  # Adjust size as needed
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image
        return img_array

    def predict(self, img_path):
        processed_image = self.preprocess_image(img_path)
        predictions = self.model.predict(processed_image)
        return predictions.argmax(axis=1)  # Return the index of the highest probability class

    def load_labels(self, labels_path):
        with open(labels_path, 'r') as f:
            labels = f.read().splitlines()
        return labels

def load_model():
    known_encodings = []
    known_names = []
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
    for name in os.listdir(DB_DIR):
        person_dir = os.path.join(DB_DIR, name)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_dir, filename)
                    img = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(img)
                    if encodings:
                        known_encodings.append(encodings[0])
                        known_names.append(name)
    return {"encodings": known_encodings, "names": known_names}

def predict_image(image_bytes, model):
    img = face_recognition.load_image_file(io.BytesIO(image_bytes))
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)
    results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(model["encodings"], face_encoding, tolerance=0.5)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = model["names"][first_match_index]
        results.append({
            "x": left,
            "y": top,
            "w": right - left,
            "h": bottom - top,
            "name": name
        })
    return {"faces": results}