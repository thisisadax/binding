import argparse
from typing import Optional, Tuple
import numpy as np

from utils import *


def place_shapes(shape1: np.ndarray, 
				 shape2: np.ndarray, 
				 shape3: Optional[np.ndarray], 
				 n_shapes: int = 10, 
				 size: int = 12) -> Image.Image:
	"""
	Place shapes on a canvas to generate a serial search trial.

	Parameters:
	shape1 (np.ndarray): The first shape image (e.g. red circle).
	shape2 (np.ndarray): The second shape image (e.g. blue triangle).
	shape3 (np.ndarray): The third (oddball) shape image. If None, it is a congruent trial. Default is None.
	n_shapes (int): The number of shapes to be placed on the canvas. Default is 10.
	size (int): The size of the shapes. Default is 12.

	Returns:
	Image.Image: The canvas with the shapes placed on it.
	"""
	# Define the canvas to draw images on, font, and drawing tool.
	canvas = np.ones((3, 256, 256), dtype=np.uint8) * 255
	canvas = np.transpose(canvas, (1, 2, 0))  # Transpose to (256x256x3) for PIL compatibility.
	canvas_img = Image.fromarray(canvas)
	# Add the shapes to the canvas.
	positions = np.zeros([n_shapes, 2])
	for i in range(n_shapes//2):
		# If it's an oddball trial and we're on the first shape, paste the oddball shape.
		if i==0 and shape3 is not None:
			positions = paste_shape(shape1, positions, canvas_img, 2*i, img_size=size)
			positions = paste_shape(shape3, positions, canvas_img, 2*i+1, img_size=size)
			continue
		positions = paste_shape(shape1, positions, canvas_img, 2*i, img_size=size)
		positions = paste_shape(shape2, positions, canvas_img, 2*i+1, img_size=size)
	return canvas_img


def make_search_trials(shape1_img: np.ndarray, 
					   shape2_img: np.ndarray, 
					   rgb1: Tuple[int, int, int], 
					   rgb2: Tuple[int, int, int], 
					   n_shapes: int = 50, size: int = 12) -> Tuple[Image.Image, Image.Image]:
	"""
	Create two trials: an congruent trial and an incongruent trial.

	Parameters:
	shape1_img (np.ndarray): The first shape image.
	shape2_img (np.ndarray): The second shape image.
	rgb1 (Tuple[int, int, int]): The RGB values for the first color.
	rgb2 (Tuple[int, int, int]): The RGB values for the second color.
	n_shapes (int): The number of shapes to be placed in each trial. Default is 50.
	size (int): The size of the shapes. Default is 12.

	Returns:
	Tuple[Image.Image, Image.Image]: A tuple containing the congruent and incongruent trials.
	"""
	s1c1 = color_shape(shape1_img.astype(np.float32), rgb1)
	s2c2 = color_shape(shape2_img.astype(np.float32), rgb2)
	s2c1 = color_shape(shape2_img.astype(np.float32), rgb1)
	s1c1 = resize(s1c1, size=size)
	s2c2 = resize(s2c2, size=size)
	s2c1 = resize(s2c1, size=size)
	congruent_trial = place_shapes(s1c1, s2c2, None, n_shapes=n_shapes, size=size)
	incongruent_trial = place_shapes(s1c1, s2c2, s2c1, n_shapes=n_shapes, size=size)
	return congruent_trial, incongruent_trial


def parse_args() -> argparse.Namespace:
	"""
	Parse command line arguments.

	Returns:
	argparse.Namespace: The parsed command line arguments.
	"""
	parser = argparse.ArgumentParser(description="Generate search trials.")
	parser.add_argument('--n_shapes', type=int, nargs='+', default=[4, 8, 12, 16, 32], help='Number of stimuli to present.')
	parser.add_argument('--n_trials', type=int, default=100, help='Number of trials to generate per condition.')
	parser.add_argument('--colors', type=str, nargs='+', default=['red', 'green', 'blue'], help='Colors to use for the shapes.')
	return parser.parse_args()


def main():
	# Parse command line arguments.
	args = parse_args()

	# Generate all possible shapes.
	imgs = np.load('imgs.npy')
	rgb_values = np.array([mcolors.to_rgb(color) for color in args.colors])
	shape_inds = np.arange(imgs.shape[0])
	rgb_inds = np.arange(rgb_values.shape[0])

	# Create directory for serial search exists.
	os.makedirs('./data/serial_search', exist_ok=True)

	# Generate the trials.
	for n in tqdm(args.n_shapes):
		for i in range(args.n_trials):
			# Randomly select an index for the first shape
			shape1_ind = np.random.choice(imgs.shape[0], size=1)[0]
			# Randomly select an index for the second shape, making sure it's not the same as the first shape
			shape2_ind = np.random.choice(shape_inds[shape_inds!=shape1_ind], size=1)[0]
			# Get the images for the selected shapes
			shape1_img = imgs[shape1_ind]
			shape2_img = imgs[shape2_ind]
			# Randomly select an index for the first color
			rgb1_ind = np.random.choice(rgb_values.shape[0], size=1)[0]
			# Randomly select an index for the second color, making sure it's not the same as the first color
			rgb2_ind = np.random.choice(rgb_inds[rgb_inds!=rgb1_ind], size=1)[0]
			# Get the RGB values for the selected colors
			rgb1 = rgb_values[rgb1_ind]
			rgb2 = rgb_values[rgb2_ind]
			# Generate the congruent and incongruent trials
			congruent_trial, incongruent_trial = make_search_trials(shape1_img, shape2_img, rgb1, rgb2, n_shapes=n, size=24)
			# Save the trials
			congruent_trial.save(f'./data/serial_search/congruent-{n}_{i}.png')
			incongruent_trial.save(f'./data/serial_search/incongruent-{n}_{i}.png')

if __name__ == '__main__':
	main()