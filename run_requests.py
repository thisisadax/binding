import os
import json
import time
import argparse
import requests
from tqdm import tqdm
from typing import Dict, Union, List
from pathlib import Path
import pandas as pd
import time
from tenacity import retry, wait_exponential, stop_after_attempt
import warnings

from utils import encode_image, get_header

warnings.filterwarnings('ignore')

@retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(5))
def run_trial(
    img_path: Union[str, List[str]],
    header: Dict[str, str],
    api_metadata: str,
    task_payload: Dict,
    parse_payload: Dict,
    parse_prompt: str,
):
    '''
    Run a trial of the serial search task.

    Parameters:
    img_path (str): The path to the image for the trial.
    header(Dict[str, str]): The API information.
    api_metadata (str): Metadata describing the relevant endpoints for the API request.
    task_payload (Dict): The payload for the vision model request.
    parse_payload (Dict): The payload for the parsing model request.
    parse_prompt (str): The prompt for the parsing model.

    Returns:
    str: The response and the parsed response from the trial.
    '''
    # Get rid of old images from the payload.
    task_payload['messages'][0]['content'] = [task_payload['messages'][0]['content'][0]]

    # If there are multiple images, encode them all.
    if isinstance(img_path, (list, tuple)):
        images = [encode_image(path) for path in img_path]
    # Otherwise, add the single image.
    else:
        images = [encode_image(img_path)]
    
    # Add the image(s) to the payload.
    image_payload = [{'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image}'}} for image in images]
    task_payload['messages'][0]['content'] += image_payload

    # Until the model provides a valid response, keep trying.
    trial_response = requests.post(api_metadata['vision_endpoint'], headers=header, json=task_payload)

    # Check for easily-avoidable errors
    if 'error' in trial_response.json():
        print('failed VLM request')
        raise ValueError('Returned error: \n' + trial_response.json()['error']['message'])
    
    # Extract the responses from the vision model and parse them with the parsing model.
    trial_response = trial_response.json()['choices'][0]['message']['content']
    trial_parse_prompt = parse_prompt + '\n' + trial_response
    parse_payload['messages'][0]['content'][0]['text'] = trial_parse_prompt # update the payload
    answer = requests.post(api_metadata['parse_endpoint'], headers=header, json=parse_payload)
    answer = answer.json()['choices'][0]['message']['content']

    # If the response is invalid raise an error.
    if 'error' in answer:
        print('failed parsing request')
        raise ValueError('Returned error: \n' + answer['error']['message'])
    elif answer=='-1':
        print('bad VLM response')
        raise ValueError(f'Invalid response: {trial_response}')
    return answer, trial_response


def parse_args() -> argparse.Namespace:
    '''
    Parse command line arguments.

    Returns:
    argparse.Namespace: The parsed command line arguments.
    '''
    parser = argparse.ArgumentParser(description='Run trials for the specified task.')
    parser.add_argument('--task_dir', type=str, required=True, help='Where the task images and metadata are stored.')
    parser.add_argument('--task_prompt_path', type=str, required=True, help='The location of the prompt file for the task.')
    parser.add_argument('--parse_prompt_path', type=str, required=True, help='The location of the prompt file for parsing the response.')
    parser.add_argument('--results_file', type=str, default=None, help='The file to save the results to.')
    parser.add_argument('--api_file', type=str, default='metadata.json', help='Location of the file containing api keys and endpoints.')
    parser.add_argument('--task_payload', type=str, default='payloads/gpt4v_single_image.json', help='The path to the task payload JSON file.')
    parser.add_argument('--parse_payload', type=str, default='payloads/gpt4_parse.json', help='The prompt for parsing the response.')
    parser.add_argument('--max_tokens', type=int, default=200, help='The maximum number of tokens for the API request.')
    parser.add_argument('--n_trials', type=int, default=None, help='The number of trials to run. Leave blank to run all trials.')
    parser.add_argument('--api', type=str, default='azure', help='Which API to use for the requests.')
    return parser.parse_args()


def main():
    # Parse command line arguments.
    args = parse_args()

    # Load the relevant payloads and prompts.
    task_payload = json.load(open(args.task_payload, 'r'))
    parse_payload = json.load(open(args.parse_payload, 'r'))
    api_metadata = json.load(open(args.api_file, 'r'))
    parse_prompt = Path(args.parse_prompt_path).read_text()
    task_prompt = Path(args.task_prompt_path).read_text()
    task_payload['messages'][0]['content'][0]['text'] = task_prompt
    task_payload['max_tokens'] = args.max_tokens

    # OpenAI API Key and header.
    header = get_header(api_metadata, model=args.api)
    api_metadata = api_metadata[args.api]

    # Load the task metadata and results.
    try:
        results_df = pd.read_csv(args.results_file)
    except FileNotFoundError:
        # If no valid results_df was provided, open the task metadata and construct a new one.
        metadata_df = pd.read_csv(os.path.join(args.task_dir, 'metadata.csv'))
        results_df = pd.DataFrame(columns=['response', 'answer'], dtype=str)
        results_df[['response', 'answer']] = ''
        results_df = pd.concat([metadata_df, results_df], axis=1)

    # Shuffle the trials, extracting n_trials if the argument was specified
    if args.n_trials:
        results_df = results_df.sample(n=args.n_trials).reset_index(drop=True)
    else:
        results_df = results_df.sample(frac=1).reset_index(drop=True)

    # Run all the trials.
    for i, trial in tqdm(results_df.iterrows()):
        # Only run the trial if it hasn't been run before.
        if type(trial.response) != str:
            try:
                answer, trial_response = run_trial(trial.path, header, api_metadata, task_payload, parse_payload, parse_prompt)
                results_df.loc[i, 'response'] = trial_response
                results_df.loc[i, 'answer'] = answer
            except Exception as e:
                print(f'Failed on trial {i} with error: {e}')
                break  # Stop the loop if there is an error and save the progress.

        if i % 10 == 0:
            # Save the results if an output file was specified, otherwise save it with the current timestamp.
            if args.results_file:
                results_df.to_csv(args.results_file, index=False)
            else:
                filename = f'results_{time.time()}.csv'
                results_df.to_csv(filename, index=False)


    # Save the results if an output file was specified, otherwise save it with the current timestamp.
    if args.results_file:
        results_df.to_csv(args.results_file, index=False)
    else:
        filename = f'results_{time.time()}.csv'
        results_df.to_csv(filename, index=False)

if __name__ == '__main__':
    main()