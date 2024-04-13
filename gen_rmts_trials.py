import os
import argparse
from functools import reduce

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import matplotlib.colors as mcolors

import torch
from torchvision.transforms import functional as F

import warnings
warnings.filterwarnings("ignore")


def make_trial(source_pair, target_pair1, target_pair2):
    # Define the canvas to draw images on, font, and drawing tool.
    canvas_img = Image.new('RGB', (512, 512), (255, 255, 255))

    # Helper function to paste images and add labels
    def paste_images_and_label(image_pair, position, label):
        img1, img2 = image_pair[0], image_pair[1]
        # Create a new image to hold the side-by-side pair
        pair_img = Image.new('RGB', (256, 256), (255, 255, 255))
        pair_img.paste(img1, (16, 96))
        pair_img.paste(img2, (128, 96))
        # Add the label below the pair image
        draw = ImageDraw.Draw(pair_img)
        font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Black.ttf', size=25)
        text_size = draw.textlength(label, font=font)
        label_position = (128 - text_size//2, 40)
        draw.text(label_position, label, (0, 0, 0), font=font)
        # Paste the pair image onto the canvas
        canvas_img.paste(pair_img, position)
        return pair_img

    # Add the image pairs to the canvas.
    source_img = paste_images_and_label(source_pair, (128, 0), 'Source Pair')
    target1_img = paste_images_and_label(target_pair1, (0, 256), 'Target Pair 1')
    target2_img = paste_images_and_label(target_pair2, (256, 256), 'Target Pair 2')

    # Convert the final canvas back to a NumPy array
    return canvas_img, source_img, target1_img, target2_img


def resize(image):
    image_array = np.transpose(image, (1, 2, 0))
    image = Image.fromarray(image_array.astype('uint8'), 'RGB')
    half_size_image = image.resize((48, 48), Image.LANCZOS)
    new_image = Image.new('RGB', (96, 96), (255, 255, 255))
    new_image.paste(half_size_image, (24, 24))
    final_image_array = np.transpose(np.array(new_image), (2, 0, 1))
    return final_image_array


def color_shape(img, rgb):
    img /= img.max()  # normalize image
    rgb /= rgb.max()  # normalize rgb code
    colored_img = (1-img) * rgb.reshape((3,1,1))
    colored_img += img
    return (colored_img * 255).astype(np.uint8)


def get_pair_inds(stim1, features, relations):
    """
    Given an input stimulus (stim1) and a target relation, retrieve the indices of stimuli that match the item along the relation.

    trial_feature: the feature to be manipulated in the trial.
    stim1: a single stimulus to start the pair.
    features: the features of all possible shapes.
    same_relation: whether the relation is same or different.
    """
    feature_names = features.columns[1:]
    feature_inds = [(features.loc[:, feat] == stim1[feat]) if feat_match else (features.loc[:, feat] != stim1[feat]) for
                    feat, feat_match in zip(feature_names, relations)]
    inds = reduce(lambda x, y: x & y, feature_inds)  # reduce to a single set of indices.
    return inds


def make_shape_trial(source1, features, source_relations, target_relations):
    """
    Make a single trial.

    trial_feature: the feature to be manipulated in the trial.
    source1: a single stimulus for the source pair.
    shape_imgs: the images of all possible shapes.
    source_relations: np.array with booleans for whether the relation matches along that feature or not.
    """
    # get the source pair
    source2_inds = get_pair_inds(source1, features, source_relations)
    source2 = features[source2_inds].sample(1).iloc[0]

    # get the correct target pair
    target_features = features[~features.ind.isin([source1.values[0], source2.values[0]])]  # exclude the source pair

    # If the task is bigger_than/smaller_than, make sure the correct target1 has the same size as source1.
    if not source_relations[-1]:
        target_features1 = target_features[
            target_features['size'] == source1['size']]  # make sure the correct target1 has the same size as source1.
        target_features1 = target_features1[(target_features1['shape'] != source1['shape']) & (
                    target_features1['color'] != source1['color'])]  # make sure the correct target1 has a different color/shape than source1.
    else:
        target_features1 = target_features[(target_features['shape'] != source1['shape']) & (
                    target_features['color'] != source1[
                'color'])]  # make sure the correct target1 has a different color/shape than source1.

    correct1 = target_features1.sample(1).iloc[0]
    correct2_inds = get_pair_inds(correct1, target_features,
                                  source_relations)  # must match all relations of source pair
    correct2 = target_features[correct2_inds].sample(1).iloc[0]

    # sample the incorrect target pair by randomly sampling a stimulus and ensuring
    # that the second one doesn't match the first along the trial relation.
    target_features = target_features[
        ~target_features.ind.isin([correct1.values[0], correct2.values[0]])]  # exclude the target pair
    incorrect1 = target_features.sample(1).iloc[0]

    # if incorrect1 has the same size as the first source and the relation is a
    # size relation, make sure that the relation is correct.
    if not source_relations[-1]:
        if incorrect1['size'] == source1['size']:
            incorrect2_inds = get_pair_inds(incorrect1, features, target_relations)
        else:
            incorrect2_inds = get_pair_inds(incorrect1, features, source_relations)
    else:
        incorrect2_inds = get_pair_inds(incorrect1, features, target_relations)
    incorrect2 = features[incorrect2_inds].sample(1).iloc[0]

    # sort the pairs by the trial type
    correct_side = np.random.rand() > 0.5
    if correct_side:
        return source1, source2, correct1, correct2, incorrect1, incorrect2, 1
    else:
        return source1, source2, incorrect2, incorrect1, correct2, correct1, 2


def get_trial_images(shape_imgs, source1, source2, correct1, correct2, incorrect1, incorrect2):
    # Get the images for the source and target pairs
    inds = np.array([source1.values[0], source2.values[0], correct1.values[0], correct2.values[0], incorrect1.values[0],
                     incorrect2.values[0]])
    trial_imgs = [F.to_pil_image(torch.tensor(img)) for img in shape_imgs[inds]]
    source_imgs, correct_target_imgs, incorrect_target_imgs = trial_imgs[:2], trial_imgs[2:4], trial_imgs[4:]

    # Make the trial randomly assigning the correct pair to either the left or right side.
    trial_img, source_pair, t1_pair, t2_pair = make_trial(
        source_imgs, correct_target_imgs, incorrect_target_imgs)
    target1_imgs, target2_imgs = correct_target_imgs, incorrect_target_imgs
    return trial_img, source_pair, t1_pair, t2_pair, source_imgs, target1_imgs, target2_imgs


def generate_rmts_trial_data():
    # Load all of the images.
    imgs = np.load('imgs.npy')

    # Plot only some test characters.
    simple_inds = [9, 98, 96, 24, 59, 51] #, 59, 55]
    simple_imgs = imgs[simple_inds]
    simple_imgs = [cv2.resize(i, (96, 96)) for i in simple_imgs]

    # Simple easily nameable colors and shapes.
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'black']
    shapes = ['triangle', 'star', 'heart', 'cross', 'pentagon', 'spade']
    rgb_values = np.array([mcolors.to_rgb(color) for color in colors])

    # Generate all possible shapes.
    all_shapes, all_features = [], []
    ind = 0
    for i, shape in enumerate(simple_imgs):
        for j, rgb in enumerate(rgb_values):
            rgb_shape = color_shape(shape.astype(np.float32), rgb)
            small_shape = resize(rgb_shape)
            all_shapes.append(small_shape)
            all_shapes.append(rgb_shape)
            all_features.append((ind, shapes[i], colors[j], 'small'))
            all_features.append((ind+1, shapes[i], colors[j], 'large'))
            ind += 2
    all_shapes = np.stack(all_shapes)
    features = pd.DataFrame(all_features, columns=['ind', 'shape', 'color', 'size'])

    # Set up metadata fields for each bit of metadata that we want to store for every trial.
    trial_cols = ['trial_type', 'correct_target', 'trial_path', 'source_path', 'target1_path', 'target2_path']
    source1_cols = ['s1_ind', 's1_shape', 's1_color', 's1_size', 's1_path']
    source2_cols = ['s2_ind', 's2_shape', 's2_color', 's2_size', 's2_path']
    target1_1_cols = ['t1_1_ind', 't1_1_shape', 't1_1_color', 't1_1_size', 't1_1_path']
    target1_2_cols = ['t1_2_ind', 't1_2_shape', 't1_2_color', 't1_2_size', 't1_2_path']
    target2_1_cols = ['t2_1_ind', 't2_1_shape', 't2_1_color', 't2_1_size', 't2_1_path']
    target2_2_cols = ['t2_2_ind', 't2_2_shape', 't2_2_color', 't2_2_size', 't2_2_path']
    col_names = (
            trial_cols + source1_cols + source2_cols + target1_1_cols
            + target1_2_cols + target2_1_cols + target2_2_cols)

    # Set up a dataframe to store all the paths for the trials that we want to save.
    same_shape_relations = [np.array([True, True, True]), np.array([False, True, True])]
    same_color_relations = [np.array([True, True, True]), np.array([True, False, True])]
    diff_size_relations = [np.array([True, True, False]), np.array([True, True, True])]
    all_relations = [same_shape_relations, same_color_relations, diff_size_relations]
    relation_names = ['sameShape', 'sameColor', 'diffSize']
    trial_df = pd.DataFrame(np.zeros([len(features)*3, len(col_names)]), columns=col_names)

    # Generate all possible trials, and save the images and metadata.
    for i, relations in enumerate(tqdm(all_relations)):
        for j, source1 in features.iterrows():
            source1, source2, target1_1, target1_2, target2_1, target2_2, correct_side = make_shape_trial(source1, features, relations[0], relations[1])
            # if source1['size'] == 'large' and source2['size'] == 'small':
            #     continue
            # if target1_1['size'] == 'large' and target1_2['size'] == 'small':
            #     continue
            # if target2_1['size'] == 'large' and target2_2['size'] == 'small':
            #     continue

            trial_img, source_pair, t1_pair, t2_pair, source_imgs, target1_imgs, target2_imgs = get_trial_images(all_shapes, source1, source2, target1_1, target1_2, target2_1, target2_2)

            # Save the relevant images.
            os.makedirs(f'data/RMTS/trial{i*len(features)+j}', exist_ok=True)
            trial_path = f'data/RMTS/trial{i*len(features)+j}/trial.png'
            source_path = f'data/RMTS/trial{i*len(features)+j}/source_pair.png'
            target1_path = f'data/RMTS/trial{i*len(features)+j}/target1_pair.png'
            target2_path = f'data/RMTS/trial{i*len(features)+j}/target2_pair.png'
            source1_path = f'data/RMTS/trial{i*len(features)+j}/source1.png'
            source2_path = f'data/RMTS/trial{i*len(features)+j}/source2.png'
            target1_1_path = f'data/RMTS/trial{i*len(features)+j}/target1_1.png'
            target1_2_path = f'data/RMTS/trial{i*len(features)+j}/target1_2.png'
            target2_1_path = f'data/RMTS/trial{i*len(features)+j}/target2_1.png'
            target2_2_path = f'data/RMTS/trial{i*len(features)+j}/target2_2.png'
            source_imgs[0].save(source1_path)
            source_imgs[1].save(source2_path)
            target1_imgs[0].save(target1_1_path)
            target1_imgs[1].save(target1_2_path)
            target2_imgs[0].save(target2_1_path)
            target2_imgs[1].save(target2_2_path)
            trial_img.save(trial_path)
            source_pair.save(source_path)
            t1_pair.save(target1_path)
            t2_pair.save(target2_path)

            trial_df.loc[i*len(features) + j, 'trial_type'] = relation_names[i]
            trial_df.loc[i*len(features) + j, 'correct_target'] = int(correct_side)
            trial_df.loc[i*len(features) + j, 'trial_path'] = trial_path
            trial_df.loc[i*len(features) + j, 'source_path'] = source_path
            trial_df.loc[i*len(features) + j, 'target1_path'] = target1_path
            trial_df.loc[i*len(features) + j, 'target2_path'] = target2_path
            trial_df.loc[i*len(features) + j, source1_cols] = list(source1.values) + [source1_path]
            trial_df.loc[i*len(features) + j, source2_cols] = list(source2.values) + [source2_path]
            trial_df.loc[i*len(features) + j, target1_1_cols] = list(target1_1.values) + [target1_1_path]
            trial_df.loc[i*len(features) + j, target1_2_cols] = list(target1_2.values) + [target1_2_path]
            trial_df.loc[i*len(features) + j, target2_1_cols] = list(target2_1.values) + [target2_1_path]
            trial_df.loc[i*len(features) + j, target2_2_cols] = list(target2_2.values) + [target2_2_path]

    trial_df.loc[~(trial_df == 0).all(axis=1)].to_csv('data/RMTS/rmts_trial_df.csv', index=False)


if __name__ == '__main__':
    generate_rmts_trial_data()


