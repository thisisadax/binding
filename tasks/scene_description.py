import os
from itertools import product, combinations_with_replacement
from typing import List
from matplotlib import colors as mcolors
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from utils import *
from tasks.task import Task, T2ITask


class SceneDescription(Task):
	
	def __init__(self,
				 n_objects: List[int],
				 n_trials: int,
				 size: int,
				 color_names: List[str],
				 shape_names: List[str],
				 shape_inds: List[int],
				 **kwargs):
		self.n_objects = n_objects
		self.n_trials = n_trials
		self.size = size
		self.color_names = color_names
		self.shape_names = shape_names
		self.shape_inds = shape_inds
		super().__init__(**kwargs)
	
	def generate_full_dataset(self):
		img_path = os.path.join(self.data_dir, 'vlm', self.task_variant, self.task_name, 'images')
		os.makedirs(img_path, exist_ok=True)
		imgs = np.load(os.path.join(self.root_dir, 'imgs.npy'))
		imgs = imgs[np.array(self.shape_inds)]
		metadata_df = pd.DataFrame(columns=['path', 'n_objects', 'n_shapes', 'n_colors', 'features', 'response', 'answer'], dtype=object)
		for n in self.n_objects:
			task_conditions = list(product(range(1, n+1), range(1, n+1)))
			condition_feature_counts = np.vstack(task_conditions).sum(axis=1)
			counts, count_freq = np.unique(condition_feature_counts, return_counts=True)
			for n_features, (n_shapes, n_colors) in tqdm(zip(condition_feature_counts, task_conditions)):
				n_trials = int(np.ceil(self.n_trials / count_freq[counts == n_features][0]))
				for i in range(n_trials):
					trial, features = self.make_trial(imgs, np.array(self.color_names), np.array(self.shape_names), n, n_shapes, n_colors, self.size)
					fname = f'nObjects={n}_nShapes={n_shapes}_nColors={n_colors}_{i}.png'
					trial_path = os.path.join(img_path, fname).split(self.root_dir+'/')[1]
					trial.save(trial_path)
					row = {'path': trial_path, 
						   'n_objects': n, 
						   'n_shapes': n_shapes, 
						   'n_colors': n_colors, 
						   'features': features}
					metadata_df = metadata_df._append(row, ignore_index=True)
		return metadata_df

	def make_trial(self, 
				   shapes: np.ndarray, 
				   color_names: np.ndarray,
				   shape_names: List[str], 
				   n_objects: int = 5, 
				   n_shapes: int = 5, 
				   n_colors: int = 5, 
				   img_size: int = 28):
		# sample the shapes and colors of objects to include in the trial.
		unique_shape_inds = np.random.choice(len(shapes), n_shapes, replace=False) # sample the unique shapes for the current trial.
		shape_inds = np.concatenate([unique_shape_inds, np.random.choice(unique_shape_inds, n_objects-n_shapes, replace=True)])
		unique_color_inds = np.random.choice(len(color_names), n_colors, replace=False)  # sample the unique colors for the current trial.
		color_inds = np.concatenate([unique_color_inds, np.random.choice(unique_color_inds, n_objects-n_colors, replace=True)])
		shape_imgs = shapes[shape_inds]
		color_names = color_names[color_inds]
		object_features = [{'shape': shape_names[shape], 'color': color} for shape, color in zip(shape_inds, color_names)]
		# recolor and resize the shapes.
		rgb_codes = np.array([mcolors.to_rgba(color)[:-1] for color in color_names]) # convert the colors to RGB format.
		colored_imgs = [color_shape(img.astype(np.float32), rgb) for img, rgb in zip(shape_imgs, rgb_codes)]
		resized_imgs = [resize(img, img_size=img_size) for img in colored_imgs]
		counting_trial = place_shapes(resized_imgs, img_size=img_size+5)
		return counting_trial, object_features

class SceneDescriptionBalanced(Task):
	def __init__(self,
				 n_objects: List[int],
				 n_trials: int,
				 size: int,
				 color_names: List[str],
				 shape_names: List[str],
				 shape_inds: List[int],
				 **kwargs):
		self.n_objects = n_objects
		self.n_trials = n_trials
		self.size = size
		self.color_names = np.array(color_names)
		self.shape_names = np.array(shape_names)
		self.shape_inds = shape_inds
		self.matrix_size = max(n_objects)
		assert self.matrix_size == len(self.color_names) == len(self.shape_names) == len(self.shape_inds)
		super().__init__(**kwargs)

	def generate_full_dataset(self) -> pd.DataFrame:
		img_path = os.path.join(self.data_dir, 'vlm', self.task_variant, self.task_name, 'images')
		os.makedirs(img_path, exist_ok=True)
		imgs = np.load(os.path.join(self.root_dir, 'imgs.npy'))
		imgs = imgs[np.array(self.shape_inds)]
		all_dfs = []
		for n in tqdm(self.n_objects):
			df = self.generate_trial_df(self.color_names, self.shape_names, n, self.n_trials)
			df['shape_inds'] = df.shape_vecs.apply(lambda x: np.where(x)[1])
			df['response'] = np.nan
			all_dfs.append(df)
			for i, trial in df.iterrows():
				trial_image = self.make_trial(imgs, trial.colors, np.array(trial.shape_inds), self.size)
				fname = f'nObjects={n}_nTriplets={int(trial.triplet_count)}_trial={i}.png'
				trial_path = os.path.join(img_path, fname).split(self.root_dir+'/')[1]
				df.loc[i, 'path'] = trial_path
				trial_image.save(trial_path)
		df = pd.concat(all_dfs, ignore_index=True)
		return df

	def make_trial(self,
				   shapes: np.ndarray, 
				   color_names: np.ndarray,
				   shape_inds: List[int], 
				   img_size: int = 28) -> Image:
		shape_imgs = shapes[shape_inds]
		rgb_codes = np.array([mcolors.to_rgba(color)[:-1] for color in color_names]) # convert the colors to RGB format.
		colored_imgs = [color_shape(img.astype(np.float32), rgb) for img, rgb in zip(shape_imgs, rgb_codes)]
		resized_imgs = [resize(img, img_size=img_size) for img in colored_imgs]
		counting_trial = place_shapes(resized_imgs, img_size=img_size+10)
		return counting_trial
	
	def generate_trial_df(self, 
						  color_names: np.ndarray, 
						  shape_names: np.ndarray, 
						  n_objects: int,
						  n_trials: int) -> pd.DataFrame:
		df = pd.DataFrame(columns=['n_shapes', 'n_colors', 'features', 'triplet_count'], dtype=object)
		task_conditions = list(product(range(1, n_objects + 1), range(1, n_objects + 1)))
		condition_feature_counts = np.vstack(task_conditions).sum(axis=1)
		counts, count_freq = np.unique(condition_feature_counts, return_counts=True)
		for n_features, (n_shapes, n_colors) in zip(condition_feature_counts, task_conditions):
			n = int(np.ceil(n_trials*10 / count_freq[counts == n_features][0]))
			for _ in range(n):
				features = self.make_trial_features(color_names, shape_names, n_objects, n_shapes, n_colors)
				trial = {'n_shapes': n_shapes, 'n_colors': n_colors, 'features': features}
				df = df._append(trial, ignore_index=True)
		df['colors'] = df.features.apply(lambda x: [obj['color'] for obj in x])
		df['shapes'] = df.features.apply(lambda x: [obj['shape'] for obj in x])
		df['color_vecs'] = df.features.apply(lambda x: np.array([color_names == obj['color'] for obj in x]).astype(int))
		df['shape_vecs'] = df.features.apply(lambda x: np.array([shape_names == obj['shape'] for obj in x]).astype(int))
		df['triplet_count'] = df.apply(lambda x: self.count_triplets(x['color_vecs'], x['shape_vecs']), axis=1)

		# Top off the number of trials for triplet counts that are less than the desired number of trials.
		triplet_counts, trial_counts = df.triplet_count.value_counts().sort_index().reset_index().values.T
		for n_triplets, n in zip(triplet_counts, trial_counts):
			if n < n_trials*10:
				valid_trials = df[df.triplet_count == n_triplets]
				n_diff = int(n_trials*10 - n)
				trial_inds = np.random.choice(valid_trials.index, n_diff, replace=True)
				cond_df = valid_trials.loc[trial_inds].copy()
				cond_df['color_vecs'] = cond_df.color_vecs.apply(lambda x: x[:, np.random.permutation(len(color_names))].astype(int).copy())
				cond_df['shape_vecs'] = cond_df.shape_vecs.apply(lambda x: x[:, np.random.permutation(len(shape_names))].astype(int).copy())
				cond_df['triplet_count'] = cond_df[['color_vecs', 'shape_vecs']].apply(lambda x: self.count_triplets(x['color_vecs'], x['shape_vecs']), axis=1)
				cond_df['colors'] = cond_df.apply(lambda x: [color_names[np.argmax(obj)] for obj in x.color_vecs], axis=1)
				cond_df['shapes'] = cond_df.apply(lambda x: [shape_names[np.argmax(obj)] for obj in x.shape_vecs], axis=1)
				cond_df['features'] = cond_df.apply(lambda x: [{'shape': shape, 'color': color} for shape, color in zip(x['shapes'], x['colors'])], axis=1)
				df = df._append(cond_df, ignore_index=True)
		df = df.groupby('triplet_count').apply(lambda x: x.sample(n_trials, replace=False)).reset_index(drop=True)
		return df
	
	def find_neighbor_inds(self, 
						   matrix: np.ndarray, 
						   i: int, 
						   j: int) -> np.ndarray:
		col_inds = np.where(matrix[i, :])[0]
		row_inds = np.where(matrix[:, j])[0]
		col_neighbor_inds = np.stack((np.repeat(i, len(col_inds)), col_inds)).T
		row_neighbor_inds = np.stack((row_inds, np.repeat(j, len(row_inds)))).T 
		return np.vstack([row_neighbor_inds, col_neighbor_inds])

	def make_trial_features(self,
							color_names: np.ndarray,
							shape_names: List[str], 
							n_objects: int = 5, 
							n_shapes: int = 5, 
							n_colors: int = 5):
		# sample the shapes and colors of objects to include in the trial.
		unique_shape_inds = np.random.choice(len(shape_names), n_shapes, replace=False)  # sample the unique shapes for the current trial.
		shape_inds = np.concatenate([unique_shape_inds, np.random.choice(unique_shape_inds, n_objects-n_shapes, replace=True)])
		unique_color_inds = np.random.choice(len(color_names), n_colors, replace=False)  # sample the unique colors for the current trial.
		color_inds = np.concatenate([unique_color_inds, np.random.choice(unique_color_inds, n_objects-n_colors, replace=True)])
		color_names = color_names[color_inds]
		object_features = [{'shape': shape_names[shape], 'color': color} for shape, color in zip(shape_inds, color_names)]
		return object_features

	def count_triplets(self, 
					x: np.ndarray, 
					y: np.ndarray) -> int:
		'''Count the number of opportunities the model has to make a binding error.
		'''
		matrix = np.einsum('tx,ty->txy', x, y).sum(0)  # [x_classes, y_classes]
		assert x.ndim == y.ndim == 2
		assert x.shape[0] == y.shape[0]
		rows, cols = matrix.shape
		triplet_count = 0
		# Iterate through each element in the matrix
		for i in range(rows):
			for j in range(cols):
				if matrix[i, j] == 0:
					continue
				# Find the neighbors to the non-zero elements
				neighbor_inds = self.find_neighbor_inds(matrix, i, j)
				# Check the neighbors for non-zero neighbors.
				for neighbor in neighbor_inds:
					if not np.allclose(neighbor, [i,j]):
						if neighbor[0] != i:
							row = matrix[neighbor[0], :] - matrix[i, :]
							row[row < 0] = 0
							mask = np.ones(len(row), dtype=bool)
							mask[j] = False # exclude the neighbor
							triplet_count += row[mask].sum()
						elif neighbor[1] != j: 
							col = matrix[:, neighbor[1]] - matrix[:, j]
							col[col < 0] = 0
							mask = np.ones(len(col), dtype=bool)
							mask[i] = False
							triplet_count += col[mask].sum()
						else:
							print('got original')
		return triplet_count 

class SceneDescriptionT2I(T2ITask):
	def __init__(self,
				 n_objects: List[int],
				 n_trials: int,
				 color_names: List[str],
				 shape_names: List[str],
				 prompt_path: str,
				 **kwargs):
		self.n_objects = n_objects
		self.n_trials = n_trials
		self.color_names = np.array(color_names)
		self.shape_names = np.array(shape_names)
		self.matrix_size = max(n_objects)
		self.prompt = Path(prompt_path).read_text()
		assert self.matrix_size == len(self.color_names) == len(self.shape_names)
		super().__init__(prompt_path=prompt_path, **kwargs)

	def generate_full_dataset(self):
		metadata_df = pd.DataFrame(columns=['path', 'prompt', 'n_objects', 'n_shapes', 'n_colors', 'features', 'completed', 'revised_prompt'])
		for n in tqdm(self.n_objects):
			# Task conditions that we want to generate trials for.
			task_conditions = list(product(range(1,n+1), range(1,n+1)))
			condition_feature_counts = np.vstack(task_conditions).sum(axis=1)
			counts, count_freq = np.unique(condition_feature_counts, return_counts=True)
			for n_features, (n_shapes, n_colors) in zip(condition_feature_counts, task_conditions):
				# Find how many trials to generate for each task condition to ensure nTrials per nFeatures condition.
				n_trials = int(np.ceil(self.n_trials / count_freq[counts==n_features][0]))
				for i in range(n_trials):
					features = self.make_trial(np.array(self.color_names), np.array(self.shape_names), n_objects=n, n_shapes=n_shapes, n_colors=n_colors)
					trial_features = [obj['color'] + ' ' + obj['shape'] for obj in features]
					objects, obj_counts = np.unique(trial_features, return_counts=True)
					objects_string = ', '.join([f'{c} {o}s' if c>1 else f'{c} {o}' for o, c in zip(objects[:-1], obj_counts[:-1])])
					objects_string += f' and {obj_counts[-1]} {objects[-1]}s' if obj_counts[-1]>1 else f', and {obj_counts[-1]} {objects[-1]}'
					trial_prompt = self.prompt.format(n_objects=n, objects_string=objects_string)
					trial_path = os.path.join(f'nObjects={n}_nShapes={n_shapes}_nColors={n_colors}_trial={i}.png')
					trial_metadata = {'path': trial_path, 
									'prompt': trial_prompt, 
									'n_objects': n, 
									'n_shapes': n_shapes, 
									'n_colors': n_colors,
									'features': features,
									'completed': False}
					metadata_df = metadata_df._append(trial_metadata, ignore_index=True)
		return metadata_df
	
	def make_trial(self, 
				   color_names: List[str],
				   shape_names: List[str], 
				   n_objects: int = 5, 
				   n_shapes: int = 5, 
				   n_colors: int = 5):
		# sample the shapes and colors of objects to include in the trial.
		unique_shape_inds = np.random.choice(len(shape_names), n_shapes, replace=False) # sample the unique shapes for the current trial.
		shape_inds = np.concatenate([unique_shape_inds, np.random.choice(unique_shape_inds, n_objects-n_shapes, replace=True)])
		unique_color_inds = np.random.choice(len(color_names), n_colors, replace=False)  # sample the unique colors for the current trial.
		color_inds = np.concatenate([unique_color_inds, np.random.choice(unique_color_inds, n_objects-n_colors, replace=True)])
		colors = color_names[color_inds]
		shapes = shape_names[shape_inds]
		object_features = [{'color': color, 'shape': shape} for color, shape in zip(colors, shapes)]
		return object_features
	
	
	def num_remaining_trials(self):
		return np.logical_not(self.results_df['completed'].astype(bool).values).sum()