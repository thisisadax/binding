import os
from typing import List
from glasbey import create_palette
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from utils import *
from pathlib import Path
import inflect
from tasks.task import Task, T2ITask


class CountingControl(Task):
	def __init__(self,
				n_objects: List[int],
				n_trials: int,
				min_size: int,
				max_size: int,
				shape_inds: List[int],
				unique_colors: bool,
				**kwargs):
		self.n_objects = n_objects
		self.shape_inds = np.array(shape_inds)
		self.n_trials = n_trials
		self.min_size = min_size
		self.max_size = max_size
		self.unique_colors = unique_colors
		super().__init__(**kwargs)

	def generate_full_dataset(self):
		img_path = os.path.join(self.data_dir, 'vlm', self.task_variant, self.task_name, 'images')
		os.makedirs(img_path, exist_ok=True)
		imgs = np.load(os.path.join(self.root_dir, 'imgs.npy'))
		imgs = imgs[self.shape_inds]
		metadata_df = pd.DataFrame(columns=['path', 'n_objects', 'response'])
		palette = create_palette(palette_size=max(self.n_objects), grid_size=256, grid_space='JCh')
		rgb_colors = np.array([mcolors.hex2color(color) for color in palette])
		for n in tqdm(self.n_objects):
			for i in range(self.n_trials):
				shape_inds = np.random.choice(len(imgs), n, replace=False)
				if self.unique_colors:
					color_inds = np.random.choice(len(rgb_colors), n, replace=False)
				else:
					color_inds = np.random.choice(len(rgb_colors), 1).repeat(n)
				colors = rgb_colors[color_inds]
				shapes = imgs[shape_inds]
				trial = self.make_trial(shapes, colors, size_range=(self.min_size, self.max_size))
				trial_path = os.path.join(img_path, f'nObjects={n}_trial={i}.png').split(self.root_dir+'/')[1]
				trial.save(trial_path)
				metadata_df = metadata_df._append({'path': trial_path, 'n_objects': n }, ignore_index=True)
		return metadata_df

	def make_trial(self, shape_imgs, colors, size_range=(10, 20)):
		sizes = np.random.randint(size_range[0], size_range[1], len(shape_imgs))
		colored_imgs = [color_shape(img.astype(np.float32), rgb) for img, rgb in zip(shape_imgs, colors)]
		resized_imgs = [resize(img, img_size=size) for img, size in zip(colored_imgs, sizes)]
		counting_trial = place_shapes(resized_imgs, img_size=max(sizes))
		return counting_trial

class Counting(Task):
	
	def __init__(self,
				 n_objects: List[int],
				 n_trials: int,
				 size: int,
				 shape_inds: List[int],
				 **kwargs):
		self.n_objects = n_objects
		self.n_trials = n_trials
		self.size = size
		self.shape_inds = shape_inds
		super().__init__(**kwargs)
	
	def generate_full_dataset(self):
		img_path = os.path.join(self.data_dir, 'vlm', self.task_variant, self.task_name, 'images')
		os.makedirs(img_path, exist_ok=True)
		imgs = np.load(os.path.join(self.root_dir, 'imgs.npy'))
		shape_img = imgs[np.array(self.shape_inds[0])]
		metadata_df = pd.DataFrame(columns=['path', 'n_objects', 'response'])
		palette = create_palette(palette_size=max(self.n_objects), grid_size=256, grid_space='JCh')
		rgb_colors = np.array([mcolors.hex2color(color) for color in palette])
		for n in tqdm(self.n_objects):
			for i in range(self.n_trials):
				if self.task_name == 'counting_high_diversity':
					color_inds = np.random.choice(len(rgb_colors), n, replace=False)
				elif self.task_name == 'counting_low_diversity':
					color_inds = np.random.choice(len(rgb_colors), 1).repeat(n)
				else:
					raise ValueError(f'Invalid task name: {self.task_name}')
				colors = rgb_colors[color_inds]
				trial = self.make_trial(shape_img, colors, n, size=self.size)
				trial_path = os.path.join(img_path, f'nObjects={n}_trial={i}.png').split(self.root_dir+'/')[1]
				trial.save(trial_path)
				metadata_df = metadata_df._append({'path': trial_path, 'n_objects': n }, ignore_index=True)
		return metadata_df

	
	def make_trial(self, img, colors, n_objects, size = (10,)):
		shape_imgs = [img for _ in range(n_objects)]
		colored_imgs = [color_shape(img.astype(np.float32), rgb) for img, rgb in zip(shape_imgs, colors)]
		resized_imgs = [resize(img, img_size=size) for img in colored_imgs]
		counting_trial = place_shapes(resized_imgs, img_size=size)
		return counting_trial


class CountingT2I(T2ITask):
	
	def __init__(self, 
				 n_objects: List[int], 
				 object_names: List[str],
				 prompt_path: str,
				 **kwargs):
		self.n_objects = n_objects
		self.object_names = object_names
		self.prompt = Path(prompt_path).read_text()
		super().__init__(prompt_path=prompt_path, **kwargs)

	def generate_full_dataset(self):
		p = inflect.engine()
		metadata_df = pd.DataFrame(columns=['path', 'prompt', 'object', 'n_objects', 'completed', 'revised_prompt'])
		for object_name in tqdm(self.object_names):
			for n in self.n_objects:
				if n>1:
					good_object_name = p.plural(object_name)
					prompt = self.prompt.format(n=n, object_name=good_object_name)
				else:
					prompt = self.prompt.format(n=n, object_name=object_name)
				trial_path = f'trial-{object_name}-N={n}.png'
				trial_metadata = {
					'path': trial_path,
					'prompt': prompt,
					'object': object_name,
					'n_objects': n,
					'completed': False}
				metadata_df = metadata_df._append(trial_metadata, ignore_index=True)
		return metadata_df
	
	def num_remaining_trials(self):
		return np.logical_not(self.results_df['completed'].astype(bool).values).sum()