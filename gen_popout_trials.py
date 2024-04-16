import argparse
from glob import glob
import os

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from utils import *


def make_popout_trials(target_img: np.array, 
					   distractor_img: np.array, 
					   rgb: tuple, 
					   n_shapes: int = 10, 
					   size: int = 20) -> tuple:
	"""
	Generate two trials. One trial contains a popout stimulus, the other does not.

	Args:
		target_img (np.array): The target image.
		distractor_img (np.array): The distractor image (if None, there is no popout stimulus).
		rgb (tuple): The RGB color for the target image.
		n_shapes (int, optional): The number of shapes. Defaults to 10.
		size (int, optional): The size of the shapes. Defaults to 20.

	Returns:
		tuple: A tuple containing the non-popout and the popout trial.
	"""
	rgb_target = color_shape(target_img.astype(np.float32), rgb)
	small_target = resize(rgb_target, size=size)
	opposite_rgb = 1 - rgb  # Calculate the opposite color
	rgb_distractor = color_shape(distractor_img.astype(np.float32), opposite_rgb)
	small_distractor = resize(rgb_distractor, size=size)
	popout_trial = place_shapes(small_target, small_distractor, n_shapes=n_shapes, img_size=size)
	uniform_trial = place_shapes(small_target, None, n_shapes=n_shapes, img_size=size)
	return uniform_trial, popout_trial


def place_shapes(source_shape: np.array, 
				 oddball_shape: np.array, 
				 n_shapes: int = 10, 
				 img_size: int = 20) -> Image:
	"""
	Places shapes on a canvas.

	Args:
		source_shape (np.array): The source shape.
		oddball_shape (np.array): The oddball shape.
		n_shapes (int, optional): The number of shapes. Defaults to 10.

	Returns:
		Image: The canvas with the shapes placed on it.
	"""
	canvas = np.ones((3, 256, 256), dtype=np.uint8) * 255
	canvas = np.transpose(canvas, (1, 2, 0))
	canvas_img = Image.fromarray(canvas)
	positions = np.zeros([n_shapes, 2])
	for i in range(n_shapes):
		if i==0 and oddball_shape is not None:
			positions = paste_shape(oddball_shape, positions, canvas_img, i, img_size=img_size)
			continue
		positions = paste_shape(source_shape, positions, canvas_img, i, img_size=img_size)
	return canvas_img


def parse_args() -> argparse.Namespace:
	"""
	Parse command line arguments.

	Returns:
	argparse.Namespace: The parsed command line arguments.
	"""
	parser = argparse.ArgumentParser(description='Generate serial search trials.')
	parser.add_argument('--n_shapes', type=int, nargs='+', default=[5,10,15,20,25,30,35,40,45,50], help='Number of stimuli to present.')
	parser.add_argument('--n_trials', type=int, default=100, help='Number of trials to generate per n_shapes condition.')
	parser.add_argument('--size', type=int, default=24, help='Size of the shapes to paste in the image.')
	parser.add_argument('--colors', type=str, nargs='+', default=None, help='Colors to use for the shapes.')
	return parser.parse_args()


def main():
	# Parse command line arguments.
	args = parse_args()

	# Load the all shapes and set up the RGB colors to use for generation.
	imgs = np.load('imgs.npy')
	if args.colors is None:
		cmap = mpl.colormaps['gist_rainbow']
		colors = cmap(np.linspace(0, 1, 100))
		rgb_values = np.array([rgba[:3]*255 for rgba in colors])
	else:
		rgb_values = np.array([mcolors.to_rgb(color) for color in args.colors])
	
	# Create directory for serial search exists.
	os.makedirs('data/popout', exist_ok=True)

	# Initialize results DataFrame for storing task performance later.
	results_df = pd.DataFrame(columns=['path', 'popout', 'n_shapes', 'response', 'answer'])

	# Generate the trials.
	for n in tqdm(args.n_shapes):
		for i in range(args.n_trials):
			# Get the images for the selected shapes
			shape1_img = imgs[37] # Circle index
			shape2_img = imgs[1]  # Diamond index
			# Generate the congruent and incongruent trials
			rgb = rgb_values[np.random.choice(rgb_values.shape[0], size=1)[0]]
			uniform_trial, popout_trial = make_popout_trials(shape1_img, shape2_img, rgb, n_shapes=n, size=args.size)
			# Save the trials and their metadata.
			uniform_trial_path = f'data/popout/uniform-{n}_{i}.png'
			uniform_trial.save(uniform_trial_path)
			results_df = results_df._append({
				'path': uniform_trial_path,
				'popout': False,
				'n_shapes': n,
				'response': None,
				'answer': None
			}, ignore_index=True)
	
			# Add the incongruent trial as a row to the DataFrame
			popout_trial_path = f'data/popout/popout-{n}_{i}.png'
			popout_trial.save(popout_trial_path)
			results_df = results_df._append({
				'path': popout_trial_path,
				'popout': True,
				'n_shapes': n,
				'response': None,
				'answer': None
			}, ignore_index=True)

	# Save results DataFrame to CSV
	results_df.to_csv('output/popout_results.csv', index=False)

if __name__ == '__main__':
	main()
