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


def make_popout_trial(shape: np.ndarray, rgb1: np.ndarray, rgb2: np.ndarray, n_objects: int = 10, img_size: int = 28) -> Image:
	# sample the shapes and colors of objects to include in the trial.
	shape_imgs = shape[np.newaxis].repeat(n_objects, axis=0)
	all_colors = rgb1.reshape(1, -1).repeat(n_objects, axis=0)
	if rgb2 is not None:
		all_colors[np.random.choice(n_objects, size=1)] = rgb2
	# recolor and resize the shapes
	colored_imgs = [color_shape(img.astype(np.float32), rgb) for img, rgb in zip(shape_imgs, all_colors)]
	resized_imgs = [resize(img, img_size=img_size) for img in colored_imgs]
	counting_trial = place_shapes(resized_imgs, img_size=img_size)
	return counting_trial

def parse_args() -> argparse.Namespace:
	"""
	Parse command line arguments.

	Returns:
	argparse.Namespace: The parsed command line arguments.
	"""
	parser = argparse.ArgumentParser(description='Generate serial search trials.')
	parser.add_argument('--n_objects', type=int, nargs='+', default=[5,10,15,20,25,30,35,40,45,50], help='Number of stimuli to present.')
	parser.add_argument('--shape_inds', type=int, nargs='+', default=[37,1], help='Indices of the shapes to use when generating the shape trials (e.g. [1,37] for diamond and circle).')
	parser.add_argument('--n_trials', type=int, default=100, help='Number of trials to generate per n_shapes condition.')
	parser.add_argument('--size', type=int, default=24, help='Size of the shapes to paste in the image.')
	parser.add_argument('--colors', type=str, nargs='+', default=None, help='Colors to use for the shapes.')
	parser.add_argument('--output_dir', type=str, default='data/popout', help='Directory to save the generated trials.')
	return parser.parse_args()


def main():
	# Parse command line arguments.
	args = parse_args()

	# Load the all shapes and set up the RGB colors to use for generation.
	imgs = np.load('data/imgs.npy')
	if args.colors is None:
		cmap = mpl.colormaps['gist_rainbow']
		colors = cmap(np.linspace(0, 1, 100))
		rgb_values = np.array([rgba[:3] for rgba in colors])
	else:
		rgb_values = np.array([mcolors.to_rgb(color) for color in args.colors])
	
	# Create directory for popout task.
	os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)

	# Initialize results DataFrame for storing task performance later.
	metadata_df = pd.DataFrame(columns=['path', 'popout', 'n_shapes'])

	# Generate the trials.
	for n in tqdm(args.n_objects):
		for i in range(args.n_trials):

			# If there are two colors, then only use those colors.
			if rgb_values.shape[0] == 2:
				rgb1 = rgb_values[0]
				rgb2 = rgb_values[1]
			else:
				rgb1 = rgb_values[np.random.choice(np.arange(rgb_values.shape[0]), size=1)]
				rgb2 = 1 - rgb1

			# Generate the congruent and incongruent trials
			shape = imgs[np.random.choice(args.shape_inds, size=1)]
			popout_trial = make_popout_trial(shape, rgb1, rgb2, n_objects=n, img_size=args.size)
			uniform_trial = make_popout_trial(shape, rgb1, None, n_objects=n, img_size=args.size)

			# Save the trials and their metadata.
			uniform_path = os.path.join(args.output_dir, 'images', f'uniform-{n}_{i}.png')
			popout_path = os.path.join(args.output_dir, 'images', f'popout-{n}_{i}.png')
			uniform_trial.save(uniform_path)
			popout_trial.save(popout_path)
			metadata_df = metadata_df._append({'path': uniform_path, 'popout': False, 'n_shapes': n}, ignore_index=True)
			metadata_df = metadata_df._append({'path': popout_path, 'popout': True, 'n_shapes': n}, ignore_index=True)

	# Save results DataFrame to CSV
	metadata_df.to_csv(os.path.join(args.output_dir, 'metadata.csv'), index=False)

if __name__ == '__main__':
	main()
