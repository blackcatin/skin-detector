import cv2
import numpy as np

# ==========================
# CHECK BLUR
# ==========================
def check_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    return round(float(blur_score), 2)

# ==========================
# CHECK BRIGHTNESS
# ==========================
def check_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    return round(float(brightness), 2)

# ==========================
# QUALITY SCORE
# ==========================
def calculate_quality_score(
    blur_score,
    brightness_score,
    target_clear_blur=400
):
    """
    Skor kualitas untuk monitoring dataset.
    Tidak digunakan sebagai filter keras.
    """

    blur_norm = min(
        np.log1p(blur_score) /
        np.log1p(target_clear_blur),
        1.0
    )

    brightness_norm = 1.0 - abs(brightness_score - 127) / 127

    final_score = (
        blur_norm * 0.6 +
        brightness_norm * 0.4
    ) * 100

    return round(final_score, 2)

# ==========================
# VALIDATION
# ==========================
def validate_image(
    image,
    min_width=100,
    min_height=100
):

    if image is None:
        return False

    height, width = image.shape[:2]

    if width < min_width:
        return False

    if height < min_height:
        return False

    # rata-rata brightness sangat rendah
    mean_pixel = np.mean(image)

    if mean_pixel < 5:
        return False

    return True

# ==========================
# HELPER FUNCTION
# ==========================
def analyze_image_quality(image):
    blur_score = check_blur(image)

    brightness_score = check_brightness(image)

    quality_score = calculate_quality_score(
        blur_score,
        brightness_score
    )

    is_valid = validate_image(image)

    return {
        "blur_score": blur_score,
        "brightness_score": brightness_score,
        "quality_score": quality_score,
        "is_valid": is_valid
    }

# ==========================
# TEST
# ==========================
if __name__ == "__main__":

    img = np.zeros(
        (600, 600, 3),
        dtype=np.uint8
    )

    result = analyze_image_quality(img)

    print(result)