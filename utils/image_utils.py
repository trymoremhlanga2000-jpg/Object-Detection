import numpy as np
import cv2
from PIL import Image


def load_image(uploaded_file):
    """
    Convert Streamlit uploaded file to OpenCV image
    """
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        return image_cv
    except Exception as e:
        raise RuntimeError(f"Image loading failed: {e}")


def convert_to_rgb(image):
    """
    Convert OpenCV image to RGB for Streamlit display
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
