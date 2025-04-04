import numpy as np
import cv2
import os
from PIL import Image


def preprocess_image(image_path):
    """
    Preprocess image for the CNN model.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Preprocessed image ready for model prediction.
    """
    try:
        # Read the image using OpenCV
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Loads as is

        # Ensure image was loaded
        if img is None:
            raise ValueError("Failed to load image. Ensure the file exists and is a valid image format.")

        # Convert grayscale to RGB
        if len(img.shape) == 2:  # Single channel (grayscale)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA (contains alpha channel)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Resize image to match model's expected input (224x224)
        img = cv2.resize(img, (224, 224))

        # Normalize pixel values to [0,1]
        img = img.astype(np.float32) / 255.0

        # Add batch dimension (required for model prediction)
        img = np.expand_dims(img, axis=0)

        return img

    except Exception as e:
        raise Exception(f"❌ Error preprocessing image: {str(e)}")
