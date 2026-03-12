import streamlit as st
import numpy as np
import cv2
from utils.detector import ObjectDetector
from utils.image_utils import load_image, convert_to_rgb


st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 YOLOv8 Object Detection App")

st.write(
    """
Upload an image and detect objects using **YOLOv8**.
"""
)


@st.cache_resource
def load_detector():
    return ObjectDetector("yolov8n.pt")


detector = load_detector()


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)


if uploaded_file:

    try:
        image = load_image(uploaded_file)

        st.subheader("Original Image")
        st.image(convert_to_rgb(image), use_container_width=True)

        if st.button("Run Detection"):

            with st.spinner("Running YOLO detection..."):

                results = detector.detect(image)

                annotated = detector.draw_boxes(image, results)

                st.subheader("Detected Objects")

                st.image(
                    convert_to_rgb(annotated),
                    use_container_width=True,
                )

                st.success("Detection completed")

    except Exception as e:
        st.error(f"Error: {e}")
