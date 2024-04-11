import os
import json

import pandas as pd
from tqdm import tqdm

from run_requests import run_trial


pair_names = ['source', 'target 1', 'target 2']

# feature decoding task parameters
feature_decoding_task = ("What is the {feature} of the {numeral} ({direction}) {pair_name} "
                         "object? Only provide the {feature}.")
features = ['shape', 'color']
numeral_direction = [('first', 'left'), ('second', 'right')]

# relation decoding task parameters
relation_decoding_task = "Are the two {pair_name} pair objects the same {relation}? Only answer this question."
relations = ['shape', 'color', 'size']

# OpenAI API Key and header.
api_key = ""
headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

# rmts task parameters
rmts_task = ("The following set of three images depict a trial of the relational match "
             "to sample task with three features: shape, color, and size. These three "
             "images depict three different pairs of objects: the source pair, target "
             "pair #1, and target pair #2. Your task is to identify which target pair "
             "shares the same abstract relations with the source pair. The possible "
             "relations are same/different shape, same/different color, and same/different "
             "size. Only one target pair will share the same relations with the source pair. "
             "Return the number of the target pair that matches the relations of the source "
             "pair and explain your reasoning.")


def run_single_rmts_trial_decomposed(info):
    """Executes one single RMTS trial (both unified + decomposed)"""
    source_image = info['source_path']
    target1_image = info['target1_path']
    target2_image = info['target2_path']

    # load the payload
    with open(os.path.join(os.path.dirname(__file__), 'payloads/gpt4v_single_image.json'), 'r') as f:
        task_payload = json.load(f)
    with open(os.path.join(os.path.dirname(__file__), 'payloads/gpt4_parse.json'), 'r') as f:
        parse_payload = json.load(f)

    # load the parsing prompts
    with open(os.path.join(os.path.dirname(__file__), 'prompts/parse_rmts_feature_decoding.txt'), 'r') as f:
        feature_decoding_prompt = f.read()
    with open(os.path.join(os.path.dirname(__file__), 'prompts/parse_rmts_relation_decoding.txt'), 'r') as f:
        relation_decoding_prompt = f.read()
    with open(os.path.join(os.path.dirname(__file__), 'prompts/parse_rmts_rmts_task.txt'), 'r') as f:
        rmts_prompt = f.read()

    # create the output data
    output_data = {
        'source_image': source_image,
        'target1_image': target1_image,
        'target2_image': target2_image,
    }

    # run all the feature decoding prompts
    for desired_pair in tqdm(pair_names):
        for desired_numeral, desired_direction in numeral_direction:
            for desired_feature in features:
                prompt = feature_decoding_task.format(
                    feature=desired_feature,
                    numeral=desired_numeral,
                    direction=desired_direction,
                    pair_name=desired_pair)

                # run the prompt
                task_payload["messages"][0]["content"][0]["text"] = prompt
                answer, trial_response = run_trial(
                    img_path=[source_image, target1_image, target2_image],
                    headers=headers,
                    task_payload=task_payload,
                    parse_payload=parse_payload,
                    parse_prompt=feature_decoding_prompt,
                    max_tokens=100 # should be a short response
                )

                # update the output data
                desired_pair_name = desired_pair.replace(' ', '')
                output_data[f'{desired_pair_name}_{desired_numeral}_{desired_feature}'] = answer
                output_data[(f'{desired_pair_name}_{desired_numeral}_'
                             f'feature_{desired_feature}_response')] = trial_response

    # run all the relation decoding prompts
    for desired_pair in tqdm(pair_names):
        for desired_relation in relations:
            prompt = relation_decoding_task.format(pair_name=desired_pair, relation=desired_relation)

            # run the prompt
            task_payload["messages"][0]["content"][0]["text"] = prompt
            answer, trial_response = run_trial(
                img_path=[source_image, target1_image, target2_image],
                headers=headers,
                task_payload=task_payload,
                parse_payload=parse_payload,
                parse_prompt=relation_decoding_prompt,
                max_tokens=100 # should be a short response
            )

            # update the output data
            desired_pair_name = desired_pair.replace(' ', '')
            output_data[f'{desired_pair_name}_{desired_relation}'] = answer
            output_data[(f'{desired_pair_name}_'
                         f'relation_{desired_relation}_response')] = trial_response

    # run the final RMTS task
    task_payload["messages"][0]["content"][0]["text"] = rmts_task
    answer, trial_response = run_trial(
        img_path=[source_image, target1_image, target2_image],
        headers=headers,
        task_payload=task_payload,
        parse_payload=parse_payload,
        parse_prompt=rmts_prompt,
        max_tokens=100  # should be a short response
    )
    output_data['rmts_answer'] = answer
    output_data['rmts_response'] = trial_response

    # save the output CSV
    return output_data


def run_single_rmts_trial_unified(info):
    """Executes one single RMTS trial (both unified + decomposed)"""
    trial_image = info['trial_path']

    # load the payload
    with open(os.path.join(os.path.dirname(__file__), 'payloads/gpt4v_single_image.json'), 'r') as f:
        task_payload = json.load(f)
    with open(os.path.join(os.path.dirname(__file__), 'payloads/gpt4_parse.json'), 'r') as f:
        parse_payload = json.load(f)

    # load the parsing prompts
    with open(os.path.join(os.path.dirname(__file__), 'prompts/parse_rmts_feature_decoding.txt'), 'r') as f:
        feature_decoding_prompt = f.read()
    with open(os.path.join(os.path.dirname(__file__), 'prompts/parse_rmts_relation_decoding.txt'), 'r') as f:
        relation_decoding_prompt = f.read()
    with open(os.path.join(os.path.dirname(__file__), 'prompts/parse_rmts_rmts_task.txt'), 'r') as f:
        rmts_prompt = f.read()

    # create the output data
    output_data = {
        'trial_image': trial_image
    }

    # run all the feature decoding prompts
    for desired_pair in tqdm(pair_names):
        for desired_numeral, desired_direction in numeral_direction:
            for desired_feature in features:
                prompt = feature_decoding_task.format(
                    feature=desired_feature,
                    numeral=desired_numeral,
                    direction=desired_direction,
                    pair_name=desired_pair)

                # run the prompt
                task_payload["messages"][0]["content"][0]["text"] = prompt
                answer, trial_response = run_trial(
                    img_path=trial_image,
                    headers=headers,
                    task_payload=task_payload,
                    parse_payload=parse_payload,
                    parse_prompt=feature_decoding_prompt,
                    max_tokens=100  # should be a short response
                )

                # update the output data
                desired_pair_name = desired_pair.replace(' ', '')
                output_data[f'{desired_pair_name}_{desired_numeral}_{desired_feature}'] = answer
                output_data[(f'{desired_pair_name}_{desired_numeral}_'
                             f'feature_{desired_feature}_response')] = trial_response

    # run all the relation decoding prompts
    for desired_pair in tqdm(pair_names):
        for desired_relation in relations:
            prompt = relation_decoding_task.format(pair_name=desired_pair, relation=desired_relation)

            # run the prompt
            task_payload["messages"][0]["content"][0]["text"] = prompt
            answer, trial_response = run_trial(
                img_path=trial_image,
                headers=headers,
                task_payload=task_payload,
                parse_payload=parse_payload,
                parse_prompt=relation_decoding_prompt,
                max_tokens=100 # should be a short response
            )

            # update the output data
            desired_pair_name = desired_pair.replace(' ', '')
            output_data[f'{desired_pair_name}_{desired_relation}'] = answer
            output_data[(f'{desired_pair_name}_'
                         f'relation_{desired_relation}_response')] = trial_response

    # run the final RMTS task
    task_payload["messages"][0]["content"][0]["text"] = rmts_task
    answer, trial_response = run_trial(
        img_path=trial_image,
        headers=headers,
        task_payload=task_payload,
        parse_payload=parse_payload,
        parse_prompt=rmts_prompt,
        max_tokens=100  # should be a short response
    )
    output_data['rmts_answer'] = answer
    output_data['rmts_response'] = trial_response

    # save the output CSV
    return output_data



