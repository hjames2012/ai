from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from PIL import Image
import io
import cv2
import face_recognition

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
    # face_recognition does not require explicit model loading
    return None

def predict_image(image_bytes, model):
    # Load image from bytes
    img = face_recognition.load_image_file(io.BytesIO(image_bytes))
    # Detect faces
    face_locations = face_recognition.face_locations(img)
    # face_locations: list of (top, right, bottom, left)
    faces_list = []
    for (top, right, bottom, left) in face_locations:
        x = left
        y = top
        w = right - left
        h = bottom - top
        faces_list.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
    return {"faces": faces_list}