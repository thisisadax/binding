import argparse
import os
import json

import matplotlib.colors as mcolors
import numpy as np
from PIL import Image

from utils import *

def make_describe_trial(imgs, scenario, n_shapes=6, n_trials=10, size=32, sigma=0.1):
    os.makedirs('data/describe', exist_ok=True)
    trials = []

    for trial_number in range(n_trials):
        # Scenario-based configuration
        if scenario == 'a':
            unique_color = False
            unique_shape = False
            shape_index = np.random.choice(len(imgs))
            color = generate_isoluminant_colors(1, mu=np.random.uniform(0, 1), sigma=sigma, uniform=True)[0]
        elif scenario == 'b':
            unique_color = False
            unique_shape = True
            shape_inds = np.random.choice(len(imgs), n_shapes, replace=False)
            color = generate_isoluminant_colors(1, mu=np.random.uniform(0, 1), sigma=sigma, uniform=True)[0]
        elif scenario == 'c':
            unique_color = True
            unique_shape = False
            shape_index = np.random.choice(len(imgs))
            colors = generate_isoluminant_colors(n_shapes, mu=np.random.uniform(0, 1), sigma=sigma, uniform=False)
        elif scenario == 'd':
            unique_color = True
            unique_shape = True
            shape_inds = np.random.choice(len(imgs), n_shapes, replace=False)
            colors = generate_isoluminant_colors(n_shapes, mu=np.random.uniform(0, 1), sigma=sigma, uniform=False)

        # Prepare shape images
        if not unique_shape:
            shape_imgs = np.array([imgs[shape_index]] * n_shapes)
            shape_ids = [shape_index] * n_shapes
        else:
            shape_imgs = imgs[shape_inds]
            shape_ids = shape_inds

        # Prepare colors
        if not unique_color:
            colors = np.array([color] * n_shapes)

        # Color and resize images
        colored_imgs = [color_shape(img.astype(np.float32), clr) for img, clr in zip(shape_imgs, colors)]
        small_imgs = [resize(img, size) for img in colored_imgs]

        # Place images on a canvas
        describe_trial, object_details = place_shapes(small_imgs, img_size=size, shape_ids=shape_ids, colors=colors)
        
        # File naming
        trial_path = f'data/describe/describe_{scenario}_{n_shapes}_{trial_number}.png'
        json_path = f'data/describe/describe_{scenario}_{n_shapes}_{trial_number}.json'
        describe_trial.save(trial_path)

        # Save JSON details
        with open(json_path, 'w') as json_file:
            json.dump(object_details, json_file, indent=4)

        trials.append({
            'image': describe_trial,
            'path': trial_path,
            'details': object_details
        })

    return trials

def place_shapes(shape_imgs, img_size, shape_ids, colors):
    # Define the canvas to draw images on and initialize positions array
    canvas = np.ones((256, 256, 3), dtype=np.uint8) * 255
    canvas_img = Image.fromarray(canvas)
    positions = np.zeros((len(shape_imgs), 2))  # Assuming this needs to be a 2D array for x, y positions
    object_details = []

    # Add the shapes to the canvas
    for i, img in enumerate(shape_imgs):
        # Update positions using paste_shape function
        positions = paste_shape(img, positions, canvas_img, i, img_size)
        # Collect details for JSON output
        object_details.append({
            'shape_id': int(shape_ids[i]),
            'color': list(colors[i])
        })

    return canvas_img, object_details

def parse_args() -> argparse.Namespace:
	"""
	Parse command line arguments.

	Returns:
	argparse.Namespace: The parsed command line arguments.
	"""
	parser = argparse.ArgumentParser(description='Generate serial search trials.')
	parser.add_argument('--n_shapes', type=int, nargs='+', default=[2,4,6,8,10], help='Number of stimuli to present.')
	parser.add_argument('--n_trials', type=int, default=100, help='Number of trials to generate per n_shapes condition.')
	parser.add_argument('--size', type=int, default=24, help='Size of the shapes to paste in the image.')
	parser.add_argument('--object_inds', type=int, nargs='+', default=[37], help='Indices of the objects to include in the trials.')
	parser.add_argument('--n_unique', type=int, default=None, help='Number of unique object shapes to include on each trial.')
	parser.add_argument('--n_colors', type=int, default=None, help='Numbe of unique colors to include on each trial')
	parser.add_argument('--uniform', type=bool, default=False, help='Whether to use uniform colors (i.e. maximally distinct)')
	parser.add_argument('--sigma', type=float, default=0.1, help='Standard deviation of the hue distribution.')
	return parser.parse_args()

def generate_isoluminant_colors(num_colors, saturation=1, lightness=0.8, mu=0.5, sigma=0.1, uniform=False):
	if uniform:
		hues = np.linspace(0, 1, num_colors, endpoint=False)
	else:
		hues = np.random.normal(loc=mu, scale=sigma, size=num_colors) % 1.0
	hsl_colors = [(hue, saturation, lightness) for hue in hues]
	rgb_colors = [mcolors.hsv_to_rgb(color) for color in hsl_colors]
	return rgb_colors

def main():
    parser = argparse.ArgumentParser(description='Generate serial search trials.')
    parser.add_argument('--scenario', type=str, choices=['a', 'b', 'c', 'd'], required=True, help='Scenario type for the trial')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials to generate')
    parser.add_argument('--n_shapes', type=int, default=6, help='Number of shapes to include in the trial')
    args = parser.parse_args()

    imgs = np.load('imgs.npy')  # Load shape images
    os.makedirs('data/describe', exist_ok=True)

    # Generate trials based on scenario and number of trials
    trials = make_describe_trial(imgs, scenario=args.scenario, n_shapes=10, n_trials=args.n_trials, size=24, sigma=0.1)
    print(f'{len(trials)} trials generated for scenario {args.scenario}')

if __name__ == '__main__':
    main()
