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
''' Plot a subset of the images
fig, axes = plt.subplots(8, 8, figsize=(7, 7), sharex=True, sharey=True, tight_layout=True)
for i, ax in enumerate(axes.flat):
    ax.imshow(imgs[i], cmap='gray')
    ax.axis('off')
'''

def make_trial(target_img, distractor_img, rgb, n_shapes=10, size=20):
    rgb_target = color_shape(target_img.astype(np.float32), rgb)
    small_target = resize(rgb_target, size=size)
    
    opposite_rgb = 255 - rgb  # Calculate the opposite color
    rgb_distractor = color_shape(distractor_img.astype(np.float32), opposite_rgb)
    small_distractor = resize(rgb_distractor, size=size)
    
    oddball_trial = place_shapes(small_target, small_distractor, n_shapes=n_shapes)
    same_trial = place_shapes(small_target, None, n_shapes=n_shapes)

    return same_trial, oddball_trial
def place_shapes(source_shape, oddball_shape, n_shapes=10):
    canvas = np.ones((3, 256, 256), dtype=np.uint8) * 255
    canvas = np.transpose(canvas, (1, 2, 0))
    canvas_img = Image.fromarray(canvas)
    
    positions = np.zeros([n_shapes, 2])
    if oddball_shape is not None:
        positions = paste_shape(oddball_shape, positions, canvas_img, 0)
        start_index = 1
    else:
        start_index = 0
    
    for i in range(start_index, n_shapes):
        positions = paste_shape(source_shape, positions, canvas_img, i)

    return canvas_img

# Helper function to paste images and add labels
def paste_shape(shape, positions, canvas_img, i, img_size = 20):
    #print(f"Shape: {shape.shape}, Positions: {positions.shape}")  # Debug print statement
    img = Image.fromarray(np.transpose(shape, (1, 2, 0)))
    position = np.array(np.random.randint(12, 244, size=2)).reshape(1,-1)
    while np.any(np.linalg.norm(positions-position, axis=1) < img_size):
        position = np.array(np.random.randint(img_size, 256-img_size, size=2)).reshape(1,-1)
    canvas_img.paste(img, tuple(position.squeeze()))
    positions[i] = position
    return positions

def color_shape(img, rgb):
    img /= img.max()  # normalize image
    rgb = rgb.astype(np.float32)  # Ensure rgb is a floating-point array before division
    rgb /= rgb.max()  # normalize rgb code
    colored_img = (1 - img) * rgb.reshape((3, 1, 1))
    colored_img += img
    return (colored_img * 255).astype(np.uint8)

def resize(image, size=12):
    image_array = np.transpose(image, (1, 2, 0))
    image = Image.fromarray(image_array.astype('uint8'), 'RGB')
    resized_image = image.resize((size, size), Image.LANCZOS)
    return np.transpose(np.array(resized_image), (2, 0, 1))

# Generate all possible shapes.
cmap = mpl.colormaps['gist_rainbow']
colors = cmap(np.linspace(0, 1, 100)) #colors = ['red', 'green', 'blue', 'purple']
rgb_values = np.array([rgba[:3]*255 for rgba in colors]) #rgb_values = np.array([mcolors.to_rgb(color) for color in colors])
shape_inds = np.arange(imgs.shape[0])
n_trials = 5
n_shapes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

for n in tqdm(n_shapes): 
    for i in range(n_trials):
        target_img = imgs[37]  # Circle index
        distractor_img = imgs[1]  # Diamond index
        #shape_inds = np.random.choice(imgs.shape[0], size=2, replace=False)
        #target_img = imgs[shape_inds[0]]  # Circle index
        #distractor_img = imgs[shape_inds[1]]  # Diamond index
        rgb = rgb_values[np.random.choice(rgb_values.shape[0], size=1)[0]]
        same_trial, oddball_trial = make_trial(target_img, distractor_img, rgb, n_shapes=n)
        same_trial.save(f'./imgs/same-{n}_{i}.png')
        oddball_trial.save(f'./imgs/diff-{n}_{i}.png')

# Specify the directory where images are saved
image_directory = '.data/popout'
# Use glob to get all the .png files in that directory
image_paths = glob(os.path.join(image_directory, '*.png')) # only get 100 trial images.

# Define a function to display a grid of images
def display_images(image_paths, rows=2, cols=5):
    fig, ax = plt.subplots(rows, cols, figsize=(15, 6))
    for i, ax in enumerate(ax.flatten()):
        if i < len(image_paths):
            img = Image.open(image_paths[i])
            ax.imshow(img)
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

# Display the first 10 images in the directory
#display_images(image_paths, rows=2, cols=5)


