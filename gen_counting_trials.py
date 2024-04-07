import base64
from glob import glob
import os
import re
import requests

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from openai import OpenAI
import pandas as pd
import plotly.express as px
from PIL import Image
#from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

# Store all 100 unicode characters.
imgs = np.load('imgs.npy')
''' #Plot a subset of the images
fig, axes = plt.subplots(8, 8, figsize=(7, 7), sharex=True, sharey=True, tight_layout=True)
for i, ax in enumerate(axes.flat):
    ax.imshow(imgs[i], cmap='gray')
    ax.axis('off')
'''

def make_trial(target_img, rgb, n_shapes=50, size=32):
    rgb_target = color_shape(target_img.astype(np.float32), rgb)
    small_target = resize(rgb_target, size=size)
    counting_trial = place_shapes(small_target, None, n_shapes=n_shapes)
    return counting_trial

def place_shapes(source_shape, oddball_shape, n_shapes=10):
    # Define the canvas to draw images on, font, and drawing tool.
    canvas = np.ones((3, 256, 256), dtype=np.uint8) * 255
    canvas = np.transpose(canvas, (1, 2, 0))  # Transpose to (256x256x3) for PIL compatibility.
    canvas_img = Image.fromarray(canvas)
    # Add the shapes to the canvas.
    positions = np.zeros([n_shapes, 2])
    for i in range(n_shapes):
        # If it's an oddball trial and we're on the first shape, paste the oddball shape.
        if i==0 and oddball_shape is not None:
            positions = paste_shape(oddball_shape, positions, canvas_img, i)
            continue
        positions = paste_shape(source_shape, positions, canvas_img, i)
    return canvas_img

# Helper function to paste images and add labels
def paste_shape(shape, positions, canvas_img, i, shape_radius=12):
    img = Image.fromarray(np.transpose(shape, (1, 2, 0)))
    max_attempts = 100  # Maximum number of attempts to find a non-overlapping position
    attempt = 0
    while attempt < max_attempts:
        position = np.random.randint(12, 244, size=2)
        # Check distance from all other shapes; assume other shapes have similar radius for simplicity
        if all(np.linalg.norm(pos - position) >= 2 * shape_radius for pos in positions if np.any(pos)):
            canvas_img.paste(img, tuple(position))
            positions[i] = position
            return positions
        attempt += 1
    raise Exception("Failed to place shape without overlap after multiple attempts.")

def color_shape(img, rgb):
    img /= img.max()  # normalize image
    if rgb.max() > 0:
        rgb /= rgb.max()  # normalize rgb code
    else: 
        pass
    colored_img = (1-img) * rgb.reshape((3,1,1))
    colored_img += img
    return (colored_img * 255).astype(np.uint8)

def resize(image, size=24):
    image_array = np.transpose(image, (1, 2, 0))
    image = Image.fromarray(image_array.astype('uint8'), 'RGB')
    resized_image = image.resize((size, size), Image.LANCZOS)
    return np.transpose(np.array(resized_image), (2, 0, 1))

n_trials = 10
n_shapes = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20]

# Generate all possible shapes.
cmap = mpl.colormaps['gist_rainbow']
#colors = cmap(np.linspace(0, 1, 100)) #colors = ['red', 'green', 'blue', 'purple']
colors = ['black']
#rgb_values = np.array([rgba[:3]*255 for rgba in colors])
rgb_values = np.array([mcolors.to_rgb(color) for color in colors])

for n in n_shapes:
    for i in range(n_trials):
        #target_ind = np.random.choice(imgs.shape[0], size=1)[0]
        target_ind = 37  # Circle index
        rgb = rgb_values[np.random.choice(rgb_values.shape[0], size=1)[0]]
        trial = make_trial(imgs[target_ind], rgb, n_shapes=n)
        trial.save(f'./data/counting/counting-{n}_{i}.png')

# Display the first 10 images in the directory
#display_images(image_paths, rows=2, cols=5)
