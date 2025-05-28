def load_image(image_path):
    from PIL import Image
    import numpy as np

    image = Image.open(image_path)
    return np.array(image)

def preprocess_image(image_array):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    image_array = image_array / 255.0  # Normalize the image
    return scaler.fit_transform(image_array.reshape(-1, image_array.shape[-1])).reshape(image_array.shape)

def validate_image(image_array):
    if image_array.ndim != 3:
        raise ValueError("Image must be a 3-dimensional array (height, width, channels)")
    if image_array.shape[2] not in [1, 3]:
        raise ValueError("Image must have 1 (grayscale) or 3 (RGB) channels")