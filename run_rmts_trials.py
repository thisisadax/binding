import os
import json
import argparse

import pandas as pd
from tqdm import tqdm

from run_requests import run_trial_azure as run_trial


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
with open("my_api_key.json", "r") as f:
    api_key = json.load(f)["azure_api_key"]
headers = {
    "Content-Type": "application/json",
    # "Authorization": f"Bearer {api_key}"
    "api-key": api_key
}

# rmts task parameters
with open(os.path.join(os.path.dirname(__file__), 'prompts/run_rmts_unified.txt'), 'r') as f:
    unified_rmts_prompt = f.read()
with open(os.path.join(os.path.dirname(__file__), 'prompts/run_rmts_decomposed.txt'), 'r') as f:
    decomposed_rmts_prompt = f.read()


def run_single_rmts_trial_decomposed(info, rmts_only=False):
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
    if not rmts_only:
        p_bar = tqdm(total=len(pair_names) * len(numeral_direction) * len(features),
                     desc="Running feature decoding prompts")
        for desired_pair in pair_names:
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
                    p_bar.update(1)
        p_bar.close()

        # run all the relation decoding prompts
        p_bar = tqdm(total=len(pair_names) * len(relations), desc="Running relation decoding prompts")
        for desired_pair in pair_names:
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
                    max_tokens=100  # should be a short response
                )

                # update the output data
                desired_pair_name = desired_pair.replace(' ', '')
                output_data[f'{desired_pair_name}_{desired_relation}'] = answer
                output_data[(f'{desired_pair_name}_'
                             f'relation_{desired_relation}_response')] = trial_response
                p_bar.update(1)
        p_bar.close()

    # run the final RMTS task
    task_payload["messages"][0]["content"][0]["text"] = decomposed_rmts_prompt
    answer, trial_response = run_trial(
        img_path=[source_image, target1_image, target2_image],
        headers=headers,
        task_payload=task_payload,
        parse_payload=parse_payload,
        parse_prompt=rmts_prompt,
        max_tokens=1000  # should be a short response
    )
    output_data['rmts_answer'] = answer
    output_data['rmts_response'] = trial_response

    # save the output CSV
    return output_data


def run_single_rmts_trial_unified(info, rmts_only=False):
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
    if not rmts_only:
        p_bar = tqdm(total=len(pair_names) * len(numeral_direction) * len(features),
                     desc="Running feature decoding prompts")
        for desired_pair in pair_names:
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
                    p_bar.update(1)
        p_bar.close()

        # run all the relation decoding prompts
        p_bar = tqdm(total=len(pair_names) * len(relations), desc="Running relation decoding prompts")
        for desired_pair in pair_names:
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
                    max_tokens=100  # should be a short response
                )

                # update the output data
                desired_pair_name = desired_pair.replace(' ', '')
                output_data[f'{desired_pair_name}_{desired_relation}'] = answer
                output_data[(f'{desired_pair_name}_'
                             f'relation_{desired_relation}_response')] = trial_response
                p_bar.update(1)
        p_bar.close()

    # run the final RMTS task
    task_payload["messages"][0]["content"][0]["text"] = unified_rmts_prompt
    answer, trial_response = run_trial(
        img_path=trial_image,
        headers=headers,
        task_payload=task_payload,
        parse_payload=parse_payload,
        parse_prompt=rmts_prompt,
        max_tokens=1000  # should be a short response
    )
    output_data['rmts_answer'] = answer
    output_data['rmts_response'] = trial_response

    # save the output CSV
    return output_data


def run_full_rmts_only_task(df):
    output_df_decomposed = pd.DataFrame()
    output_df_unified = pd.DataFrame()
    df['correct_target'] = df['correct_target'].astype(int)

    # for each row, run the RMTS task and get the output
    try:
        for i, row in (pbar := tqdm(df.iterrows(), total=len(df), desc="Running RMTS task")):
            output_data_decomposed = run_single_rmts_trial_decomposed(row, rmts_only=True)
            output_data_unified = run_single_rmts_trial_unified(row, rmts_only=True)
            output_data_decomposed['correct_target'] = row['correct_target']
            output_data_unified['correct_target'] = row['correct_target']

            if i == 0:
                output_df_decomposed = pd.DataFrame(output_data_decomposed, index=[0])
                output_df_unified = pd.DataFrame(output_data_unified, index=[0])
            else:
                output_df_decomposed.loc[i] = output_data_decomposed
                output_df_unified.loc[i] = output_data_unified

            # Compute the accuracy of each task
            output_df_decomposed['rmts_answer'] = output_df_decomposed['rmts_answer'].astype(int)
            output_df_unified['rmts_answer'] = output_df_unified['rmts_answer'].astype(int)
            decomposed_correct = (output_df_decomposed['rmts_answer'] == output_df_decomposed['correct_target']).sum()
            unified_correct = (output_df_unified['rmts_answer'] == output_df_unified['correct_target']).sum()
            pbar.set_postfix({"decomposed_acc": {decomposed_correct / len(output_df_decomposed)},
                              "unified_acc": {unified_correct / len(output_df_unified)}})
    except KeyboardInterrupt as k:
        print("Completing Trial")
        pass

    # Compute the accuracy of each task
    output_df_decomposed['rmts_answer'] = output_df_decomposed['rmts_answer'].astype(int)
    output_df_unified['rmts_answer'] = output_df_unified['rmts_answer'].astype(int)
    decomposed_correct = (output_df_decomposed['rmts_answer'] == output_df_decomposed['correct_target']).sum()
    unified_correct = (output_df_unified['rmts_answer'] == output_df_unified['correct_target']).sum()
    print(f"Decomposed accuracy: {decomposed_correct / len(output_df_decomposed)} (correct = {decomposed_correct})")
    print(f"Unified accuracy: {unified_correct / len(output_df_unified)} (correct = {unified_correct})")

    # Save the output
    output_df_decomposed.to_csv('data/RMTS/rmts_output_decomposed.csv', index=False)
    output_df_unified.to_csv('data/RMTS/rmts_output_unified.csv', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', type=str, default='rmts_only',
                    choices=['workspace', 'rmts_only'])
    args = ap.parse_args()

    # load the trial dataframe
    if args.task == 'workspace':
        df = pd.read_csv('data/RMTS/rmts_trial_df.csv')
        output_decomposed = pd.DataFrame()
        output_unified = pd.DataFrame()

        # iterate over every row in the dataframe
        for i, row in df.iterrows():
            # currently not enough time
            if i == 5:
                break

            # run the trial
            output_data_decomposed = run_single_rmts_trial_decomposed(row)
            output_data_unified = run_single_rmts_trial_unified(row)
            print(output_data_decomposed)
            print(output_data_unified)

            # save the output
            if i == 0:
                output_decomposed = pd.DataFrame(output_data_decomposed, index=[0])
                output_unified = pd.DataFrame(output_data_unified, index=[0])
            else:
                output_decomposed.loc[i] = output_data_decomposed
                output_unified.loc[i] = output_data_unified

            # check the accuracy
            print(
                "Row {i}: Correct = {correct}, Decomposed = {decomposed}, Unified = {unified}".format(
                    i=i,
                    correct=row['correct_target'],
                    decomposed=output_data_decomposed['rmts_answer'],
                    unified=output_data_unified['rmts_answer']
                )
            )

        output_decomposed.to_csv('data/RMTS/rmts_trial_output_decomposed.csv', index=False)
        output_unified.to_csv('data/RMTS/rmts_trial_output_unified.csv', index=False)
    else:
        run_full_rmts_only_task(pd.read_csv('data/RMTS/rmts_trial_df.csv'))


