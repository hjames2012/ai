# test_face_recognition.py
import face_recognition

img = face_recognition.load_image_file("test.jpg")  # Use a local image with a clear face
faces = face_recognition.face_locations(img)
print(faces)