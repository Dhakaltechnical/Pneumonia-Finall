from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to the image
image_path = "chest_xray/val/NORMAL/IM-0115-0001.jpeg"

def is_chest_xray(image_path):
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found or unable to read.")

    height, width, _ = image.shape
    aspect_ratio = width / height

    if height < 200 or width < 200 or not (1.0 < aspect_ratio < 1.6):
        return aspect_ratio

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_gray_value = np.mean(gray_image)
    
    return mean_gray_value

# Example usage
aspect_ratio_or_mean_gray = is_chest_xray(image_path)
print(aspect_ratio_or_mean_gray)
