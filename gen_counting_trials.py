import argparse
import os

import matplotlib.colors as mcolors
import numpy as np
from PIL import Image

from utils import *


def make_counting_trial(imgs, n_shapes=10, n_unique=5, size=32, uniform=False, sigma=0.1):
	# sample the shapes to include in the trial.
	unique_inds = np.random.choice(len(imgs), n_unique, replace=False)
	shape_inds = np.random.choice(unique_inds, n_shapes, replace=True)
	shape_imgs = imgs[shape_inds]
	# color the shapes
	mu = np.random.uniform(0,1)
	colors = generate_isoluminant_colors(n_shapes, mu=mu, sigma=sigma, uniform=uniform)
	colored_imgs = [color_shape(img.astype(np.float32), rgb) for img, rgb in zip(shape_imgs, colors)]
	small_imgs = [resize(img, size=size) for img in colored_imgs]
	counting_trial = place_shapes(small_imgs, img_size=size)
	return counting_trial

def place_shapes(shape_imgs, img_size=32):
	# Define the canvas to draw images on, font, and drawing tool.
	canvas = np.ones((3, 256, 256), dtype=np.uint8) * 255 #204
	canvas = np.transpose(canvas, (1, 2, 0))  # Transpose to (256x256x3) for PIL compatibility.
	canvas_img = Image.fromarray(canvas)
	# Add the shapes to the canvas.
	n_shapes = len(shape_imgs)
	positions = np.zeros([n_shapes, 2])
	for i, img in enumerate(shape_imgs):
		positions = paste_shape(img, positions, canvas_img, i, img_size=img_size)
	return canvas_img

def generate_isoluminant_colors(num_colors, saturation=1, lightness=0.8, mu=0.5, sigma=0.1, uniform=False):
	if uniform:
		hues = np.linspace(0, 1, num_colors, endpoint=False)
	else:
		hues = np.random.normal(loc=mu, scale=sigma, size=num_colors) % 1.0
	hsl_colors = [(hue, saturation, lightness) for hue in hues]
	rgb_colors = [mcolors.hsv_to_rgb(color) for color in hsl_colors]
	return rgb_colors

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
	parser.add_argument('--uniform', type=bool, default=False, help='Whether to use uniform colors (i.e. maximally distinct)')
	parser.add_argument('--sigma', type=float, default=0.1, help='Standard deviation of the hue distribution.')
	return parser.parse_args()

def main():
	# Parse command line arguments.
	args = parse_args()
	assert args.n_unique <= len(args.object_inds), 'Number of unique objects must be less than or equal to the number of objects.'
	imgs = np.load('imgs.npy')
	imgs = imgs[np.array(args.object_inds)]  # sample only the shapes that we want to include in the trials.

	# Create directory for serial search exists.
	os.makedirs('data/counting', exist_ok=True)

	# Initialize results DataFrame for storing task performance later.
	results_df = pd.DataFrame(columns=['path', 'n_shapes', 'response', 'answer'])

	# Generate the trials.
	for n in args.n_shapes:
		for i in range(args.n_trials):
			trial = make_counting_trial(imgs, n_shapes=n, n_unique=args.n_unique, size=args.size, uniform=args.uniform, sigma=args.sigma)

			# Save the trials and their metadata.
			trial_path = f'data/counting/counting-{n}_{i}.png'
			trial.save(trial_path)
			trial.save(trial_path)
			results_df = results_df._append({
				'path': trial_path,
				'incongruent': False,
				'n_shapes': n,
				'response': None,
				'answer': None
			}, ignore_index=True)

	# Save results DataFrame to CSV
	results_df.to_csv('./output/counting_results.csv', index=False)

if __name__ == '__main__':
	main()