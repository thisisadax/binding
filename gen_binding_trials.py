import argparse
import os
import json
from typing import List, Tuple

import matplotlib.colors as mcolors
import numpy as np
from PIL import Image
import pandas as pd

from utils import paste_shape, color_shape, resize, place_shapes


def make_binding_trial(shapes: np.ndarray, 
					   colors: np.ndarray,
					   shape_names: List[str], 
					   n_objects: int = 5, 
					   n_shapes: int = 5, 
					   n_colors: int = 5, 
					   img_size: int = 28):
	# sample the shapes and colors of objects to include in the trial.
	unique_shape_inds = np.random.choice(len(shapes), n_shapes, replace=False) # sample the unique shapes for the current trial.
	shape_inds = np.concatenate([unique_shape_inds, np.random.choice(unique_shape_inds, n_objects-n_shapes, replace=True)])
	unique_color_inds = np.random.choice(len(colors), n_colors, replace=False)  # sample the unique colors for the current trial.
	color_inds = np.concatenate([unique_color_inds, np.random.choice(unique_color_inds, n_objects-n_colors, replace=True)])
	shape_imgs = shapes[shape_inds]
	colors = colors[color_inds]
	object_features = [{'shape': shape_names[shape], 'color': color} for shape, color in zip(shape_inds, colors)]
	# recolor and resize the shapes
	rgb_codes = np.array([mcolors.to_rgba(color)[:-1] for color in colors]) # convert the colors to RGB format.
	colored_imgs = [color_shape(img.astype(np.float32), rgb) for img, rgb in zip(shape_imgs, rgb_codes)]
	resized_imgs = [resize(img, img_size=img_size) for img in colored_imgs]
	counting_trial = place_shapes(resized_imgs, img_size=img_size+10)
	return counting_trial, object_features

def parse_args() -> argparse.Namespace:
	"""
	Parse command line arguments.

	Returns:
	argparse.Namespace: The parsed command line arguments.
	"""
	parser = argparse.ArgumentParser(description='Generate feature binding trials.')
	parser.add_argument('--n_objects', type=int, nargs='+', default=[2,3,4,5,6,7,8], help='Number of stimuli to present.')
	parser.add_argument('--n_trials', type=int, default=100, help='Number of trials to generate per n_shapes condition.')
	parser.add_argument('--size', type=int, default=24, help='Size of the shapes to paste in the image.')
	parser.add_argument('--color_names', type=str, nargs='+', default=['red', 'green', 'blue', 'gold', 'purple', 'orange', 'saddlebrown', 'pink', 'gray', 'black'], help='Colors to use for the shapes.')
	parser.add_argument('--shape_names', type=str, nargs='+', default=['triangle', 'cloud', 'cross', 'down arrow', 'umbrella', 'pentagon', 'heart', 'star'], help='Names of the shapes to use in the trials.')
	parser.add_argument('--shape_inds', type=int, nargs='+', default=[9,21,24,28,34,59,96,98], help='Indices of the shapes to include in the trials.')
	parser.add_argument('--output_dir', type=str, default='data/binding', help='Directory to save the generated trials.')
	return parser.parse_args()

def main():
	# Load shape images and trial configuration.
	args = parse_args()
	imgs = np.load('data/imgs.npy')
	imgs = imgs[args.shape_inds]
	assert len(args.shape_names) == len(args.shape_inds) 
	assert len(args.color_names) == len(args.shape_names) 

	# Create directory for binding task.
	os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)

	# Initialize DataFrame for storing task metadata_df later.
	metadata_df = pd.DataFrame(columns=['path', 'n_objects', 'n_shapes', 'n_colors', 'features', 'shapes_names', 'color_names'], dtype=object)

	# Generate the trials.
	for n in args.n_objects:

		# Task conditions that we want to generate trials for.
		n2 = int(np.ceil(n/2))
		task_conditions = [(1,1), (1,n), (n,1), (n,n), (n2,n2)]

		# Generate n_trials for each task condition.
		for i in range(args.n_trials):

			# Generate trials for each task condition.
			for n_shapes, n_colors in task_conditions:

				# Save the trials and their metadata.
				trial, features = make_binding_trial(imgs, np.array(args.color_names), n_objects=n, n_shapes=n_shapes, n_colors=n_colors, img_size=args.size, shape_names=args.shape_names)
				fname = f'nObjects={n}_nShapes={n_shapes}_nColors={n_colors}_{i}.png'
				trial_path = os.path.join(args.output_dir, 'images', fname)
				trial.save(trial_path)
				row = {'path': trial_path, 
		   			   'n_objects': n, 
					   'n_shapes': n_shapes, 
					   'n_colors': n_colors, 
					   'features': features,
					   'shapes_names': args.shape_names,
					   'color_names': args.color_names}
				metadata_df = metadata_df._append(row, ignore_index=True)

	# Save results DataFrame to CSV.
	metadata_df.to_csv(os.path.join(args.output_dir, 'metadata.csv'), index=False)

if __name__ == '__main__':
	main()
