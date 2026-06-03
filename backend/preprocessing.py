import cv2
import os
import numpy as np

def denoise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)


def clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = c.apply(l)

    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def resize(image, size=(224, 224)):
    return cv2.resize(image, size)

def preprocess_pipeline(image: np.ndarray) -> np.ndarray:
    image = denoise(image)
    image = clahe(image)
    image = resize(image)
    return image

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Image corrupt")
    return img


def save_image(image, output_path):
    cv2.imwrite(output_path, image)

def preprocess_image(input_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    image = load_image(input_path)
    processed = preprocess_pipeline(image)

    filename = os.path.basename(input_path)
    output_path = os.path.join(output_folder, filename)

    save_image(processed, output_path)

    return output_path