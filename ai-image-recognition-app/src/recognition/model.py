from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from PIL import Image
import io
import cv2

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
    # Load OpenCV's pre-trained Haar Cascade for face detection
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    return face_cascade

def predict_image(image_bytes, model):
    # Decode image bytes to numpy array
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return {"faces": []}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # faces: list of (x, y, w, h)
    faces_list = []
    for (x, y, w, h) in faces:
        faces_list.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
    return {"faces": faces_list}