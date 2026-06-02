import os
import cv2
import numpy as np


# ==========================
# LOAD IMAGE
# ==========================

def load_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Gagal membaca gambar: {image_path}")

    return image


# ==========================
# DENOISING
# ==========================

def apply_denoising(image):
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        5,
        5,
        7,
        21
    )


# ==========================
# CLAHE
# ==========================

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    merged = cv2.merge((l, a, b))

    return cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2BGR
    )


# ==========================
# RESIZE
# ==========================

def resize_image(image, size=(224, 224)):
    return cv2.resize(image, size)

# ==========================
# SAVE IMAGE
# ==========================

def save_processed_image(image, output_path):
    cv2.imwrite(output_path, image)


# ==========================
# MAIN PIPELINE
# ==========================

def preprocess_image(
    input_path,
    processed_folder="uploads/processed"
):
    
    os.makedirs(processed_folder, exist_ok=True)

    image = load_image(input_path)

    image = apply_denoising(image)

    image = apply_clahe(image)

    image = resize_image(image)

    filename = os.path.basename(input_path)

    output_path = os.path.join(
        processed_folder,
        filename
    )

    save_processed_image(
        image,
        output_path
    )

    return output_path