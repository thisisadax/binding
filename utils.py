import base64
from glob import glob
import os
import re
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from openai import OpenAI
import pandas as pd
from PIL import Image
from tqdm import tqdm
import time


def paste_shape(shape: np.ndarray, 
                positions: np.ndarray, 
                canvas_img: Image.Image, 
                i: int, 
                img_size: int = 12) -> np.ndarray:
    """
    Paste a shape onto a canvas image at a random position.

    Parameters:
    shape (np.ndarray): The shape to be pasted.
    positions (np.ndarray): The positions of the shapes on the canvas.
    canvas_img (Image.Image): The canvas image.
    i (int): The index of the current shape.
    img_size (int): The size of the shape. Default is 12.

    Returns:
    np.ndarray: The updated positions of the shapes on the canvas.
    """
    img = Image.fromarray(np.transpose(shape, (1, 2, 0)))
    position = np.array(np.random.randint(img_size, 256-img_size, size=2)).reshape(1,-1)
    # Keep trying to find a position that is far enough from the other shapes.
    while np.any(np.linalg.norm(positions-position, axis=1) < img_size):
        position = np.array(np.random.randint(img_size, 256-img_size, size=2)).reshape(1,-1)
    canvas_img.paste(img, tuple(position.squeeze()))
    positions[i] = position
    return positions


def color_shape(img: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """
    Color a grayscale image with a given RGB code.

    Parameters:
    img (np.ndarray): The grayscale image.
    rgb (np.ndarray): The RGB code.

    Returns:
    np.ndarray: The colored image.
    """
    img /= img.max()  # normalize image
    rgb /= rgb.max()  # normalize rgb code
    colored_img = (1-img) * rgb.reshape((3,1,1))
    colored_img += img
    return (colored_img * 255).astype(np.uint8)


def resize(image: np.ndarray, size: int = 12) -> np.ndarray:
    """
    Resize an image to a given size.

    Parameters:
    image (np.ndarray): The image to be resized.
    size (int): The size to resize the image to. Default is 12.

    Returns:
    np.ndarray: The resized image.
    """
    image_array = np.transpose(image, (1, 2, 0))
    image = Image.fromarray(image_array.astype('uint8'), 'RGB')
    resized_image = image.resize((size, size), Image.LANCZOS)
    return np.transpose(np.array(resized_image), (2, 0, 1))


def encode_image(image_path):
    """
    Encode an image as a base64 string.

    Parameters:
    image_path (str): The path to the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')