import argparse
from functools import reduce
import os
from tqdm import tqdm
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import sys
from utils import *
from tasks.task import Task

class RMTS(Task):
	def __init__(self, n_trials, size, color_names, shape_names, shape_inds, font_path, condition, subtask, **kwargs):
		self.n_trials = n_trials
		self.size = size
		self.color_names = color_names
		self.shape_names = shape_names
		self.shape_inds = shape_inds
		self.font_path = font_path 
		self.condition = condition
		self.subtask = subtask
		super().__init__(**kwargs)
	
	def generate_full_dataset(self):
		img_path = os.path.join(self.data_dir, 'vlm', self.task_variant, self.task_name, 'images')
		os.makedirs(img_path, exist_ok=True)
		imgs = np.load(os.path.join(self.root_dir, 'imgs.npy'))
		imgs = imgs[np.array(self.shape_inds)]
		rgb_values = np.array([mcolors.to_rgb(color) for color in self.color_names])
		object_df, object_features, all_objects = self.gen_all_shapes(imgs, rgb_values, self.color_names)
		pair_df, pair_shapes = self.gen_all_pairs(object_features, all_objects)
		trial_metadata = []
		features = ['color', 'shape']
		pairs = [('source', 'top'), ('target1', 'bottom left'), ('target2', 'bottom right')]
		objects = [(1, '1st', 'left'), (2, '2nd', 'right')]
		feature_task_columns = ['unified_path', 'decomposed_paths', 'feature', 'feature_value', 'pair', 'pair_loc', 'object_loc', 'object_ind', 'response']
		relation_task_columns = ['unified_path', 'decomposed_paths', 'relation', 'relation_value', 'pair', 'pair_loc', 'response']
		full_task_columns = ['unified_path', 'decomposed_paths', 'correct', 'response']
		feature_task_df = pd.DataFrame(np.zeros([self.n_trials, len(feature_task_columns)]), columns=feature_task_columns, dtype=object)
		relation_task_df = pd.DataFrame(np.zeros([self.n_trials, len(relation_task_columns)]), columns=relation_task_columns, dtype=object)
		full_task_df = pd.DataFrame(np.zeros([self.n_trials, len(full_task_columns)]), columns=full_task_columns, dtype=object)
		for n in tqdm(range(self.n_trials)):
			value_counts = pair_df.same_object.value_counts()
			probs = value_counts / value_counts.sum()
			source_probs = np.zeros(len(pair_df))
			source_probs[pair_df.same_object == True] = 0.5 / probs[True]
			source_probs[pair_df.same_object == False] = 0.5 / probs[False]
			source_probs = source_probs / source_probs.sum()
			source_ind = np.random.choice(len(pair_df), p=source_probs)
			feature_inds = np.where(pair_df.loc[(source_ind, ['same_shape', 'same_color'])].values)[0]
			feature = features[np.random.choice(feature_inds)]
			trial_features, source_pair, target_pair1, target_pair2 = self.make_rmts_trial(source_ind, pair_df, pair_shapes, relation=feature)
			canvas_img, source_img, target1_img, target2_img = self.make_trial(source_pair, target_pair1, target_pair2)
			trial_metadata.append(trial_features)
			unified_path = os.path.join(img_path, f'trial-{n}.png').split(self.root_dir+'/')[1]
			source_path = os.path.join(img_path, f'source-{n}.png').split(self.root_dir+'/')[1]
			target1_path = os.path.join(img_path, f'target1-{n}.png').split(self.root_dir+'/')[1]
			target2_path = os.path.join(img_path, f'target2-{n}.png').split(self.root_dir+'/')[1]
			decomposed_paths = [source_path, target1_path, target2_path]
			canvas_img.save(unified_path)
			source_img.save(source_path)
			target1_img.save(target1_path)
			target2_img.save(target2_path)
			feature = features[np.random.choice(len(features))]
			pair, pair_loc = pairs[np.random.choice(len(pairs))]
			obj_id, obj_index, obj_loc = objects[np.random.choice(len(objects))]
			full_task_df.loc[n,:] = [unified_path, str(decomposed_paths), trial_features.correct_target[0], np.nan]
			feature_value = trial_features[f'{pair}_{feature}{obj_id}'].values[0]
			relation_value = trial_features[f'{pair}_same_{feature}'].values[0]
			feature_task_df.loc[n,:] = [unified_path, str(decomposed_paths), feature, feature_value, pair, pair_loc, obj_loc, obj_index, np.nan]
			relation_task_df.loc[n,:] = [unified_path, str(decomposed_paths), f'same_{feature}', relation_value, pair, pair_loc, np.nan]
		task_path = os.path.join(self.data_dir, 'vlm', self.task_variant, self.task_name)
		trial_metadata_df = pd.concat(trial_metadata, axis=0)
		trial_metadata_df.to_csv(os.path.join(task_path, 'trial_metadata.csv'), index=False)
		feature_task_df.to_csv(os.path.join(task_path, 'feature_task_metadata.csv'), index=False)
		relation_task_df.to_csv(os.path.join(task_path, 'relation_task_metadata.csv'), index=False)
		full_task_df.to_csv(os.path.join(task_path, 'full_task_metadata.csv'), index=False)
		trial = trial_metadata_df.apply(self.get_trial_json, axis=1).reset_index()
		feature_task2_df = full_task_df[['unified_path', 'decomposed_paths', 'response']]
		feature_task2_df['features'] = trial[0].values.copy()
		feature_task2_df.to_csv(os.path.join(task_path, 'feature2_task_metadata.csv'), index=False)
		return self.get_task_metadata(self.subtask, feature_task_df, feature_task2_df, relation_task_df, full_task_df)

	def make_rmts_trial(self, source_ind: int, feature_df: pd.DataFrame, pair_imgs: np.array, relation: str='color'):
		source_features = feature_df.iloc[feature_df.index == source_ind].reset_index(drop=True)
		source_imgs = pair_imgs[source_ind]
		source_relations = source_features[['same_shape', 'same_color']]
		inds = [feature_df[i] == j for i, j in zip(source_relations.columns, source_relations.values[0])]
		correct_inds = reduce((lambda x, y: x & y), inds)
		correct_target_features = feature_df[correct_inds & (feature_df.index != source_ind)].sample(1)
		correct_target_imgs = pair_imgs[correct_target_features.index[0]].squeeze()
		color_mask = feature_df.same_color == source_relations.same_color.values[0]
		shape_mask = feature_df.same_shape == source_relations.same_shape.values[0]
		if relation == 'color':
			incorrect_inds = ~color_mask & shape_mask
		elif relation == 'shape':
			incorrect_inds = color_mask & ~shape_mask
		else:
			raise ValueError('Relation must be "color" or "shape"')
		incorrect_target_features = feature_df[incorrect_inds].sample(1)
		incorrect_target_imgs = pair_imgs[incorrect_target_features.index[0]].squeeze()
		correct_side = np.random.rand() > 0.5
		if correct_side:
			target1_features = correct_target_features.add_prefix('target1_').reset_index(drop=True)
			target2_features = incorrect_target_features.add_prefix('target2_').reset_index(drop=True)
			target1_imgs = correct_target_imgs
			target2_imgs = incorrect_target_imgs
			correct_target = 1
		else:
			target1_features = incorrect_target_features.add_prefix('target1_').reset_index(drop=True)
			target2_features = correct_target_features.add_prefix('target2_').reset_index(drop=True)
			target1_imgs = incorrect_target_imgs
			target2_imgs = correct_target_imgs
			correct_target = 2
		source_features = source_features.add_prefix('source_').reset_index(drop=True)
		trial_features = pd.concat([source_features, target1_features, target2_features], axis=1)
		trial_features['correct_target'] = correct_target
		trial_features['task_relation'] = f'same_{relation}'
		return (trial_features, source_imgs, target1_imgs, target2_imgs)
	
	def gen_all_pairs(self, object_features, all_objects):
		feat1 = np.repeat(object_features, len(object_features), axis=0)
		feat2 = np.tile(object_features, (len(object_features), 1))
		shapes1 = np.repeat(all_objects, len(all_objects), axis=0)
		shapes2 = np.tile(all_objects, (len(all_objects), 1, 1, 1))
		pair_features = np.hstack([feat1,feat2])
		pair_df = pd.DataFrame(pair_features, columns=['shape1', 'color1', 'shape2', 'color2'])
		pair_df['same_color'] = pair_df['color1'] == pair_df['color2']
		pair_df['same_shape'] = pair_df['shape1'] == pair_df['shape2']
		pair_df['same_object'] = pair_df['same_color'].values & pair_df['same_shape'].values
		good_inds = pair_df['same_color'] | pair_df['same_shape']
		pair_df = pair_df[good_inds].reset_index(drop=True)
		pair_shapes = np.stack([shapes1[good_inds], shapes2[good_inds]], axis=1)
		return (pair_df, pair_shapes)

	def gen_all_shapes(self, imgs, rgb_values, colors):
		all_features = []
		all_objects = []
		for i, img in enumerate(imgs):
			for j, rgb in enumerate(rgb_values):
				rgb_img = color_shape(img.astype(np.float32), rgb)
				all_features.append([self.shape_names[i], colors[j]])
				all_objects.append(rgb_img)
		object_df = pd.DataFrame(all_features, columns=['shape', 'color'])
		object_features = np.array(all_features)
		all_objects = np.stack(all_objects)
		return (object_df, object_features, all_objects)

	def get_trial_json(self, row):
		pairs = ['source', 'target1', 'target2']
		objects = ['1', '2']
		trial = {}
		for pair in pairs:
			trial[pair] = {}
			for obj in objects:
				trial[pair][f'{pair}_object{obj}'] = {'shape': row[f'{pair}_shape{obj}'], 
													  'color': row[f'{pair}_color{obj}']}
		return trial

	
	def make_trial(self, source_pair, target_pair1, target_pair2):
		canvas_img = Image.new('RGB', (512, 512), (255, 255, 255))
		
		def paste_images_and_label(image_pair, position, label):
			img1 = image_pair[0]
			img2 = image_pair[1]
			img1 = resize(img1.astype(np.uint8), 96)
			img2 = resize(img2.astype(np.uint8), 96)
			img1 = Image.fromarray(np.transpose(img1, (1, 2, 0)))
			img2 = Image.fromarray(np.transpose(img2, (1, 2, 0)))
			pair_img = Image.new('RGB', (256, 256), (255, 255, 255))
			pair_img.paste(img1, (16, 96))
			pair_img.paste(img2, (128, 96))
			draw = ImageDraw.Draw(pair_img)
			font = ImageFont.truetype(self.font_path, size=self.size)
			text_size = draw.textlength(label, font=font)
			label_position = (128 - text_size // 2, 40)
			draw.text(label_position, label, (0, 0, 0), font=font)
			canvas_img.paste(pair_img, position)
			return pair_img

		source_img = paste_images_and_label(source_pair, (128, 0), 'Source Pair')
		target1_img = paste_images_and_label(target_pair1, (0, 256), 'Target Pair 1')
		target2_img = paste_images_and_label(target_pair2, (256, 256), 'Target Pair 2')
		return canvas_img, source_img, target1_img, target2_img
	
	def get_task_metadata(self, subtask_name, feature_task_df, feature_task2_df, relation_task_df, full_task_df):
		if subtask_name == 'features':
			return feature_task_df
		elif subtask_name == 'features2':
			return feature_task2_df
		elif subtask_name == 'relations':
			return relation_task_df
		elif subtask_name == 'full':
			return full_task_df
		else:
			raise ValueError('Subtask name must be "features", "features2", "relations", or "full"')