import argparse
import matplotlib.colors as mcolors
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from utils import *


def letter_img(letter: str):
	assert len(letter)==1 # make sure the string is just a letter.
	img = Image.new('RGB', (32, 32), (255, 255, 255))
	font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Black.ttf', size=28)
	draw = ImageDraw.Draw(img)
	draw.text((7, -4), letter, (0,0,0), font=font)
	img_array = np.transpose(np.array(img), (2, 0, 1))
	return img_array


def make_search_trial(shape1: np.ndarray, shape2: np.ndarray, rgb1: np.ndarray, rgb2: np.ndarray, n_objects: int = 10, oddball: bool = True, img_size: int = 28) -> Image:
	objects = [(shape1, rgb1), (shape2, rgb2)]
	# Add the oddball object first.
	if oddball:
		all_shapes = [shape1]
		all_colors = [rgb2]
		n_objects -= 1
	else:
		all_shapes = []
		all_colors = []
	for i in range(n_objects): #int(np.ceil(n_objects/2))
		random_index = np.random.choice(len(objects))
		all_shapes.append(objects[random_index][0])
		all_colors.append(objects[random_index][1])
	# recolor and resize the shapes
	colored_imgs = [color_shape(img.astype(np.float32), rgb) for img, rgb in zip(all_shapes, all_colors)]
	resized_imgs = np.stack([resize(img, img_size=img_size) for img in colored_imgs])
	np.random.shuffle(resized_imgs) # shuffle the order of the images list
	counting_trial = place_shapes(resized_imgs, img_size=img_size+5) # make shapes a little further apart
	return counting_trial


def parse_args() -> argparse.Namespace:
	"""
	Parse command line arguments.

	Returns:
	argparse.Namespace: The parsed command line arguments.
	"""
	parser = argparse.ArgumentParser(description='Generate serial search trials.')
	parser.add_argument('--n_objects', type=int, nargs='+', default=[4, 6, 8, 10, 16, 32], help='Number of stimuli to present.')
	parser.add_argument('--n_trials', type=int, default=100, help='Number of trials to generate per n_shapes condition.')
	parser.add_argument('--colors', type=str, nargs='+', default=None, help='Colors to use for the shapes.')
	parser.add_argument('--shape_inds', type=int, nargs='+', default=None, help='Indices of the shapes to use when generating the shape trials (e.g. [1,37] for diamond and circle).')
	parser.add_argument('--size', type=int, default=28, help='Size to reshape the shapes to.')
	parser.add_argument('--use_letters', type=bool, default=False, help='Whether or not to use letters as stimuli.')
	parser.add_argument('--output_dir', type=str, default='data/binding', help='Directory to save the generated trials.')
	return parser.parse_args()


def main():
	# Parse command line arguments.
	args = parse_args()

	# Load the shapes to generate trials with. 
	if args.shape_inds:
		assert args.use_letters=='False'
		imgs = np.load('data/imgs.npy')
		shape_inds = np.array(args.shape_inds) 
	elif args.use_letters:
		img1 = letter_img('L')
		img2 = letter_img('T')
		imgs = np.stack([img1, img2])
		shape_inds = np.array([0, 1])
	else:
		raise ValueError('Either shape_inds or use_letters must be specified.')

	# Set up the colors to use when generating the stimuli.
	if args.colors is None:
		cmap = mpl.colormaps['gist_rainbow']
		colors = cmap(np.linspace(0, 1, 100))
		rgb_values = np.array([rgba[:3] for rgba in colors])
	else:
		rgb_values = np.array([mcolors.to_rgb(color) for color in args.colors])

	# Create directory for serial search exists.
	os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)

	# Initialize results DataFrame for storing task performance later.
	metadata_df = pd.DataFrame(columns=['path', 'incongruent', 'n_shapes'])

	# Generate the trials.
	for n in tqdm(args.n_objects):
		for i in range(args.n_trials):
			# Only randomly select the shapes if we're not using letters.
			if args.use_letters:
				shape1 = imgs[0]
				shape2 = imgs[1]
			else:
				# Randomly select the two shapes.
				shape1_ind = np.random.choice(shape_inds, size=1)[0]
				shape2_ind = np.random.choice(shape_inds[shape_inds!=shape1_ind], size=1)[0]
				shape1 = imgs[shape1_ind]
				shape2 = imgs[shape2_ind]

			# Only randomly select the colors if we have more than two colors.
			if rgb_values.shape[0]==2:
				rgb1 = rgb_values[0]
				rgb2 = rgb_values[1]
			elif args.colors:
				# Randomly select the two colors is a list of colors is provided.
				rgb1_ind = np.random.choice(rgb_values.shape[0], size=1)[0]
				rgb2_options = rgb_values[rgb_values!=rgb_values[rgb1_ind]]
				rgb2_ind = np.random.choice(rgb2_options, size=1)[0]
				rgb1 = rgb_values[rgb1_ind]
				rgb2 = rgb_values[rgb2_ind]
			else:
				# Select the colors to be opposites if we're using the rainbow colormap.
				rgb1_ind = np.random.choice(rgb_values.shape[0], size=1)[0]
				rgb1 = rgb_values[rgb1_ind]
				rgb2 = 1-rgb1
			# Generate the congruent and incongruent trials
			congruent_trial = make_search_trial(shape1, shape2, rgb1, rgb2, n_objects=n, oddball=False, img_size=args.size)
			incongruent_trial = make_search_trial(shape1, shape2, rgb1, rgb2, n_objects=n, oddball=True, img_size=args.size)
			# Save the trials and their metadata.
			congruent_path = os.path.join(args.output_dir, 'images', f'congruent-{n}_{i}.png')
			incongruent_path = os.path.join(args.output_dir, 'images', f'incongruent-{n}_{i}.png')
			congruent_trial.save(congruent_path)
			incongruent_trial.save(incongruent_path)
			metadata_df = metadata_df._append({'path': congruent_path, 'incongruent': False, 'n_shapes': n}, ignore_index=True)
			metadata_df = metadata_df._append({'path': incongruent_path, 'incongruent': True, 'n_shapes': n}, ignore_index=True)

	# Save results DataFrame to CSV
	metadata_df.to_csv(os.path.join(args.output_dir, 'metadata.csv'), index=False)

if __name__ == '__main__':
	main()