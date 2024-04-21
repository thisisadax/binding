import os
import copy
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

np.random.seed(0)


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
    rgb /= (rgb.max() + 1e-6)  # normalize rgb code
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
    feature_inds = [(features.loc[:, feat] == stim1[feat]) if feat_match else (features.loc[:, feat] != stim1[feat]) for feat, feat_match in zip(feature_names, relations)]
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
        target_features1 = target_features[target_features['size'] == source1['size']]  # make sure the correct target1 has the same size as source1.
        target_features1 = target_features1[(target_features1['shape'] != source1['shape']) & (target_features1['color'] != source1['color'])]  # make sure the correct target1 has a different color/shape than source1.
    else:
        target_features1 = target_features[(target_features['shape'] != source1['shape']) & (target_features['color'] != source1['color'])]  # make sure the correct target1 has a different color/shape than source1.
    correct1 = target_features1.sample(1).iloc[0]
    correct2_inds = get_pair_inds(correct1, target_features, source_relations)  # must match all relations of source pair
    correct2 = target_features[correct2_inds].sample(1).iloc[0]

    # sample the incorrect target pair by randomly sampling a stimulus and ensuring
    # that the second one doesn't match the first along the trial relation.
    target_features = target_features[~target_features.ind.isin([correct1.values[0], correct2.values[0]])]  # exclude the target pair
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
    trial_imgs = [F.to_pil_image(
        torch.tensor(
            shape_imgs[(i['shape'], i['color'], i['size'])]
        )
    ) for i in [source1, source2, correct1, correct2, incorrect1, incorrect2]]

    # stack all images in a row and show
    trial_img = Image.new('RGB', (768 * 2, 256), (255, 255, 255))
    for i, img in enumerate(trial_imgs):
        trial_img.paste(img, (256*i, 0))

    source_imgs, correct_target_imgs, incorrect_target_imgs = trial_imgs[:2], trial_imgs[2:4], trial_imgs[4:]

    # Make the trial randomly assigning the correct pair to either the left or right side.
    trial_img, source_pair, t1_pair, t2_pair = make_trial(source_imgs, correct_target_imgs, incorrect_target_imgs)
    target1_imgs, target2_imgs = correct_target_imgs, incorrect_target_imgs
    return trial_img, source_pair, t1_pair, t2_pair, source_imgs, target1_imgs, target2_imgs


def generate_all_rmts_trial_data():
    # Load all the images.
    imgs = np.load('imgs.npy')

    # Plot only some test characters.
    simple_inds = [9, 98, 96, 24, 100, 101]
    simple_imgs = imgs[simple_inds]
    simple_imgs = [cv2.resize(i, (96, 96)) for i in simple_imgs]

    # Simple easily nameable colors and shapes.
    colors = ['red', 'green', 'blue', 'purple', 'saddlebrown', 'black']
    shapes = ['triangle', 'star', 'heart', 'cross', 'circle', 'square']
    rgb_values = np.array([mcolors.to_rgb(color) for color in colors])

    # Generate all possible shapes.
    all_features = []
    all_shape_feature_dict = {}
    ind = 0
    for i, shape in enumerate(simple_imgs):
        for j, rgb in enumerate(rgb_values):
            rgb_shape = color_shape(shape.astype(np.float32), rgb)
            small_shape = resize(rgb_shape)
            all_shape_feature_dict[(shapes[i], colors[j], 'small')] = small_shape
            all_shape_feature_dict[(shapes[i], colors[j], 'large')] = rgb_shape
            all_features.append((ind, shapes[i], colors[j], 'small'))
            all_features.append((ind + 1, shapes[i], colors[j], 'large'))
            ind += 2
    features = pd.DataFrame(all_features, columns=['ind', 'shape', 'color', 'size'])

    # Generate two separate directories for the unified and decomposed tasks.
    os.makedirs('data/unified_RMTS', exist_ok=True)
    os.makedirs('data/decomposed_RMTS', exist_ok=True)

    # get all the possible types of trials
    same_shape_relations = [np.array([True, True, True]), np.array([False, True, True])]
    same_color_relations = [np.array([True, True, True]), np.array([True, False, True])]
    diff_size_relations = [np.array([True, True, False]), np.array([True, True, True])]
    all_relations = [same_shape_relations, same_color_relations, diff_size_relations]

    # Generate all possible trials, and save the images and metadata.
    trial_count = 0
    feature_decoding_task_dict_unified = {
        "path": [],
        "feature": [],
        "feature_value": [],
        "pair": [],
        "pair_loc": [],
        "object_loc": [],
        "object_ind": []
    }
    feature_decoding_decomposed_path_list = []
    relation_decoding_task_dict_unified = {
        "path": [],
        "relation": [],
        "relation_value": [],
        "pair": [],
        "pair_loc": [],
    }
    relation_decoding_decomposed_path_list = []
    rmts_task_dict_unified = {
        "path": [],
        "correct": [],
    }
    rmts_task_decomposed_path_list = []

    for i, relations in enumerate(tqdm(all_relations)):
        for j, source1 in features.iterrows():
            # generate the trial info (shapes, sizes, relations)
            # skip trials where the left shape is larger than the right shape (for any)
            source1, source2, target1_1, target1_2, target2_1, target2_2, correct_side = (
                make_shape_trial(source1, features, relations[0], relations[1]))
            for (img1, img2) in [[source1['size'], source2['size']],
                                 [target1_1['size'], target1_2['size']],
                                 [target2_1['size'], target2_2['size']]]:
                if img1 == 'large' and img2 == 'small':
                    continue

            # generate feature decoding task information
            loc_information_dict = [[source1.to_dict(), {"pair": "source", "pair_loc": "top", "object_loc": "left", "object_ind": 1}],
                                    [source2.to_dict(), {"pair": "source", "pair_loc": "top", "object_loc": "right", "object_ind": 2}],
                                    [target1_1.to_dict(), {"pair": "target1", "pair_loc": "bottom-left", "object_loc": "left", "object_ind": 1}],
                                    [target1_2.to_dict(), {"pair": "target1", "pair_loc": "bottom-left", "object_loc": "right", "object_ind": 2}],
                                    [target2_1.to_dict(), {"pair": "target2", "pair_loc": "bottom-right", "object_loc": "left", "object_ind": 1}],
                                    [target2_2.to_dict(), {"pair": "target2", "pair_loc": "bottom-right", "object_loc": "right", "object_ind": 2}]]
            for img, loc_info in loc_information_dict:
                feature_decoding_task_dict_unified["path"].append(f"data/unified_RMTS/trial{str(trial_count).zfill(3)}_unified.png")
                feature_decoding_task_dict_unified["feature"].append("shape")
                feature_decoding_task_dict_unified["feature_value"].append(img["shape"])
                feature_decoding_task_dict_unified["pair"].append(loc_info["pair"])
                feature_decoding_task_dict_unified["pair_loc"].append(loc_info["pair_loc"])
                feature_decoding_task_dict_unified["object_loc"].append(loc_info["object_loc"])
                feature_decoding_task_dict_unified["object_ind"].append(loc_info["object_ind"])
                feature_decoding_decomposed_path_list.append(
                    [f"data/unified_RMTS/trial{str(trial_count).zfill(3)}_decomposed_source.png",
                        f"data/unified_RMTS/trial{str(trial_count).zfill(3)}_decomposed_target1.png",
                        f"data/unified_RMTS/trial{str(trial_count).zfill(3)}_decomposed_target2.png"])
            for img, loc_info in loc_information_dict:  # repeat the shape task but for colors
                feature_decoding_task_dict_unified["path"].append(f"data/unified_RMTS/trial{str(trial_count).zfill(3)}_unified.png")
                feature_decoding_task_dict_unified["feature"].append("color")
                feature_decoding_task_dict_unified["feature_value"].append(img["color"])
                feature_decoding_task_dict_unified["pair"].append(loc_info["pair"])
                feature_decoding_task_dict_unified["pair_loc"].append(loc_info["pair_loc"])
                feature_decoding_task_dict_unified["object_loc"].append(loc_info["object_loc"])
                feature_decoding_task_dict_unified["object_ind"].append(loc_info["object_ind"])
                feature_decoding_decomposed_path_list.append(
                    [f"data/unified_RMTS/trial{str(trial_count).zfill(3)}_decomposed_source.png",
                        f"data/unified_RMTS/trial{str(trial_count).zfill(3)}_decomposed_target1.png",
                        f"data/unified_RMTS/trial{str(trial_count).zfill(3)}_decomposed_target2.png"])

            # create the relation decoding task information
            relation_loc_info = [[source1.to_dict(), source2.to_dict(),
                                  {"pair": "source", "pair_loc": "top", "pair_idx": 0}],
                                 [target1_1.to_dict(), target1_2.to_dict(),
                                  {"pair": "target1", "pair_loc": "bottom-left", "pair_idx": 1}],
                                 [target2_1.to_dict(), target2_2.to_dict(),
                                  {"pair": "target2", "pair_loc": "bottom-right", "pair_idx": 2}]]
            for img1, img2, loc_info in relation_loc_info:
                relation = "same" if img1["size"] == img2["size"] else "different"
                relation_decoding_task_dict_unified["path"].append(f"data/unified_RMTS/trial{str(trial_count).zfill(3)}_unified.png")
                relation_decoding_task_dict_unified["relation"].append("size")
                relation_decoding_task_dict_unified["relation_value"].append(relation)
                relation_decoding_task_dict_unified["pair"].append(loc_info["pair"])
                relation_decoding_task_dict_unified["pair_loc"].append(loc_info["pair_loc"])
                relation_decoding_decomposed_path_list.append(
                    [f"data/unified_RMTS/trial{str(trial_count).zfill(3)}_decomposed_source.png",
                        f"data/unified_RMTS/trial{str(trial_count).zfill(3)}_decomposed_target1.png",
                        f"data/unified_RMTS/trial{str(trial_count).zfill(3)}_decomposed_target2.png"])
            for img1, img2, loc_info in relation_loc_info:
                relation = "same" if img1["shape"] == img2["shape"] else "different"
                relation_decoding_task_dict_unified["path"].append(f"data/unified_RMTS/trial{str(trial_count).zfill(3)}_unified.png")
                relation_decoding_task_dict_unified["relation"].append("shape")
                relation_decoding_task_dict_unified["relation_value"].append(relation)
                relation_decoding_task_dict_unified["pair"].append(loc_info["pair"])
                relation_decoding_task_dict_unified["pair_loc"].append(loc_info["pair_loc"])
                relation_decoding_decomposed_path_list.append(
                    [f"data/unified_RMTS/trial{str(trial_count).zfill(3)}_decomposed_source.png",
                        f"data/unified_RMTS/trial{str(trial_count).zfill(3)}_decomposed_target1.png",
                        f"data/unified_RMTS/trial{str(trial_count).zfill(3)}_decomposed_target2.png"])
            for img1, img2, loc_info in relation_loc_info:
                relation = "same" if img1["color"] == img2["color"] else "different"
                relation_decoding_task_dict_unified["path"].append(f"data/unified_RMTS/trial{str(trial_count).zfill(3)}_unified.png")
                relation_decoding_task_dict_unified["relation"].append("color")
                relation_decoding_task_dict_unified["relation_value"].append(relation)
                relation_decoding_task_dict_unified["pair"].append(loc_info["pair"])
                relation_decoding_task_dict_unified["pair_loc"].append(loc_info["pair_loc"])
                relation_decoding_decomposed_path_list.append(
                    [f"data/unified_RMTS/trial{str(trial_count).zfill(3)}_decomposed_source.png",
                        f"data/unified_RMTS/trial{str(trial_count).zfill(3)}_decomposed_target1.png",
                        f"data/unified_RMTS/trial{str(trial_count).zfill(3)}_decomposed_target2.png"])

            # generate the RMTS task information
            rmts_task_dict_unified["path"].append(f"data/unified_RMTS/trial{str(trial_count).zfill(3)}_unified.png")
            rmts_task_dict_unified["correct"].append(correct_side)
            rmts_task_decomposed_path_list.append(
                [f"data/unified_RMTS/trial{str(trial_count).zfill(3)}_decomposed_source.png",
                    f"data/unified_RMTS/trial{str(trial_count).zfill(3)}_decomposed_target1.png",
                    f"data/unified_RMTS/trial{str(trial_count).zfill(3)}_decomposed_target2.png"])

            # Save the relevant images
            trial_img, source_pair, t1_pair, t2_pair, source_imgs, target1_imgs, target2_imgs = (
                get_trial_images(all_shape_feature_dict, source1, source2,
                                 target1_1, target1_2, target2_1, target2_2))
            trial_num = str(trial_count).zfill(3)
            trial_path = f'data/unified_RMTS/trial{trial_num}_unified.png'
            source_path = f'data/decomposed_RMTS/trial{trial_num}_decomposed_source.png'
            target1_path = f'data/decomposed_RMTS/trial{trial_num}_decomposed_target1.png'
            target2_path = f'data/decomposed_RMTS/trial{trial_num}_decomposed_target2.png'
            trial_img.save(trial_path)
            source_pair.save(source_path)
            t1_pair.save(target1_path)
            t2_pair.save(target2_path)
            trial_count += 1

    # update the list of paths for the decomposed task
    feature_decoding_task_dict_decomposed = copy.deepcopy(feature_decoding_task_dict_unified)
    feature_decoding_task_dict_decomposed["path"] = feature_decoding_decomposed_path_list
    relation_decoding_task_dict_decomposed = copy.deepcopy(relation_decoding_task_dict_unified)
    relation_decoding_task_dict_decomposed["path"] = relation_decoding_decomposed_path_list
    rmts_task_dict_decomposed = copy.deepcopy(rmts_task_dict_unified)
    rmts_task_dict_decomposed["path"] = rmts_task_decomposed_path_list

    # save the CSVs for the tasks
    feature_decoding_task_df_unified = pd.DataFrame(feature_decoding_task_dict_unified)
    feature_decoding_task_df_unified.to_csv('data/unified_RMTS/feature_decoding_task_unified.csv', index=False)
    feature_decoding_task_df_decomposed = pd.DataFrame(feature_decoding_task_dict_decomposed)
    feature_decoding_task_df_decomposed.to_csv('data/decomposed_RMTS/feature_decoding_task_decomposed.csv', index=False)
    relation_decoding_task_df_unified = pd.DataFrame(relation_decoding_task_dict_unified)
    relation_decoding_task_df_unified.to_csv('data/unified_RMTS/relation_decoding_task_unified.csv', index=False)
    relation_decoding_task_df_decomposed = pd.DataFrame(relation_decoding_task_dict_decomposed)
    relation_decoding_task_df_decomposed.to_csv('data/decomposed_RMTS/relation_decoding_task_decomposed.csv', index=False)
    rmts_task_df_unified = pd.DataFrame(rmts_task_dict_unified)
    rmts_task_df_unified.to_csv('data/unified_RMTS/rmts_task_unified.csv', index=False)
    rmts_task_df_decomposed = pd.DataFrame(rmts_task_dict_decomposed)
    rmts_task_df_decomposed.to_csv('data/decomposed_RMTS/rmts_task_decomposed.csv', index=False)


if __name__ == '__main__':
    generate_all_rmts_trial_data()


