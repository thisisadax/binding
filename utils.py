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
                img_size: int = 40) -> np.ndarray:
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
    position = np.array(np.random.randint(0, 256-img_size, size=2)).reshape(1,-1)
    # Keep trying to find a position that is far enough from the other shapes.
    while np.any(np.linalg.norm(positions-position, axis=1) < img_size):
        position = np.array(np.random.randint(0, 256-img_size, size=2)).reshape(1,-1)
    canvas_img.paste(img, tuple(position.squeeze()))
    positions[i] = position
    return positions


def color_shape(img: np.ndarray, rgb: np.ndarray, bg_color: float = 1, all_black: bool = False) -> np.ndarray:
    """
    Color a grayscale image with a given RGB code.

    Parameters:
    img (np.ndarray): The grayscale image.
    rgb (np.ndarray): The RGB code.
    bg_color (float): The background color. Default is 1.
    all_black (bool): Whether to color the image black. Default is False.

    Returns:
    np.ndarray: The colored image.
    """
    if all_black:
        rgb = np.ones(3)
        return img.astype(np.uint8) * rgb.reshape((3,1,1))
    # Normalize the RGB code.
    rgb = rgb.astype(np.float32)
    if rgb.max() > 1:
        rgb /= rgb.max()  # normalize rgb code
    img /= img.max()  # normalize image
    colored_img = (1-img) * rgb.reshape((3,1,1))
    colored_img += img * bg_color
    return (colored_img * 255).astype(np.uint8)


def resize(image: np.ndarray, img_size: int=28) -> np.ndarray:
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
    resized_image = image.resize((img_size, img_size), Image.LANCZOS)
    return np.transpose(np.array(resized_image), (2, 0, 1))


def encode_image(image_path):
    """
    Encode an image as a base64 string.

    Parameters:
    image_path (str): The path to the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')