from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from PIL import Image
import io
import cv2
import face_recognition
import os
from skimage import transform as trans

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
                    aligned_face = align_face(img)  # Align the face
                    if aligned_face is not None:
                        encodings = face_recognition.face_encodings(aligned_face)
                        if encodings:
                            known_encodings.append(encodings[0])
                            known_names.append(name)
                        else:
                            print(f"No encoding found for {img_path}") # Add this line
                    else:
                        print(f"Alignment failed for {img_path}") # Add this line
    return {"encodings": known_encodings, "names": known_names}

def align_face(img):
    face_locations = face_recognition.face_locations(img)
    if not face_locations:
        print("No face found in align_face")  # Add this line
        return None

    face_landmarks_list = face_recognition.face_landmarks(img, face_locations=face_locations)
    if not face_landmarks_list:
        print("No landmarks found in align_face")  # Add this line
        return None

    face_landmarks = face_landmarks_list[0]  # Assuming only one face

    # Define the desired left and right eye positions.
    desired_left_eye = (0.35, 0.35)
    desired_right_eye = (0.65, 0.35)
    desired_face_width = 200
    desired_face_height = desired_face_width

    # Get the actual eye positions
    left_eye = np.mean(face_landmarks['left_eye'], axis=0)
    right_eye = np.mean(face_landmarks['right_eye'], axis=0)

    # Compute the angle between the eye centroids
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # Compute the scale
    desired_dist = (desired_right_eye[0] - desired_left_eye[0]) * desired_face_width
    actual_dist = np.sqrt((dX ** 2) + (dY ** 2))
    scale = desired_dist / actual_dist

    # Compute the center of the face
    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    # Compute the transformation matrix
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    # Update the translation component of the matrix
    tX = desired_face_width * 0.5 - eyes_center[0]
    tY = desired_face_height * desired_left_eye[1] - eyes_center[1]
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])

    # Apply the affine transformation
    (w, h) = (desired_face_width, desired_face_height)
    output = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

    return output

def predict_image(image_bytes, model):
    img = face_recognition.load_image_file(io.BytesIO(image_bytes))
    aligned_face = align_face(img)
    if aligned_face is None:
        print("No aligned face found in predict_image") # Add this line
        return {"faces": []}

    face_locations = face_recognition.face_locations(aligned_face)
    face_encodings = face_recognition.face_encodings(aligned_face, face_locations)
    results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(model["encodings"], face_encoding, tolerance=0.5)
        name = "Unknown"
        confidence = 0.0  # Initialize confidence

        if True in matches:
            first_match_index = matches.index(True)
            name = model["names"][first_match_index]

            # Calculate a simple confidence score (you can improve this)
            face_distances = face_recognition.face_distance(model["encodings"], face_encoding)
            confidence = 1.0 - face_distances[first_match_index]

        results.append({
            "x": left,
            "y": top,
            "w": right - left,
            "h": bottom - top,
            "name": name,
            "confidence": float(confidence)  # Ensure it's serializable
        })
    return {"faces": results}