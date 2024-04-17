import re
import json
import time
import argparse
import requests
from tqdm import tqdm
from typing import Dict, Any, Union, List
from pathlib import Path
import numpy as np
import pandas as pd
import time

from utils import encode_image


def get_header(api_info, model='gpt-4-azure') -> Dict[str, str]:
    if model == 'gpt-4-vision-azure':
        return {
            "Content-Type": "application/json",
            "api-key": f"{api_info['azure_api_key']}"
        }
    if model == 'gpt-4':
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_info['openai_api_key']}"
        }
    # TODO: Add Claude Opus and Google Gemeni endpoints as well.
    else: 
        raise ValueError(f"Model {model} not recognized.")


def run_trial(
    img_path: Union[str, List[str]],
    api_info: Dict[str, str],
    task_payload: Dict,
    parse_payload: Dict,
    parse_prompt: str,
    n_attempts: int = 5,
    max_tokens: int = 200,
):
    """
    Run a trial of the serial search task.

    Parameters:
    img_path (str): The path to the image for the trial.
    api_info(Dict[str, str]): The API information.
    task_payload (Dict): The payload for the vision model request.
    parse_payload (Dict): The payload for the parsing model request.
    parse_prompt (str): The prompt for the parsing model.
    n_attempts (int): The number of attempts to make. Default is 5.
    max_tokens (int): The maximum number of tokens for the API request. Default is 200.

    Returns:
    str: The response and the parsed response from the trial.
    """
    # Update the max number of tokens.
    task_payload["max_tokens"] = max_tokens

    # Encode the image and update the task payload.
    if isinstance(img_path, (list, tuple)):
        image = encode_image(img_path[0])
        task_payload["messages"][0]["content"][1]["image_url"]["url"] = f"data:image/jpeg;base64,{image}"

        for i, path in enumerate(img_path[1:]):
            image = encode_image(path)
            task_payload["messages"][0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}"
                }
            })
    else:
        image = encode_image(img_path)
        task_payload["messages"][0]["content"][1]["image_url"]["url"] = f"data:image/jpeg;base64,{image}"

    # Get the relevant headers
    headers = get_header(api_info, model='gpt-4')

    # Until the model provides a valid response, keep trying.
    answer = "-1"  # -1 if model failed to provide a valid response.
    i = 0  # Counter to keep track of the number of attempts. Limit to n_attempts.
    trial_response = None
    while answer == "-1" and i < n_attempts:
        # Get the vision model response.
        trial_response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=task_payload,
        )
        try:
            trial_response = trial_response.json()["choices"][0]["message"]["content"]
        except KeyError as e:
            # Check for API key errors.
            if 'API key' in trial_response.json()['error']['message']:
                raise ValueError("API key is invalid.")
            # If we've hit a rate limit, then wait and try again.
            if 'error' in trial_response.json():
                # possibly encountered an error
                if 'Please try again in' not in trial_response.json()['error']['message']:
                    n_attempts += 1
                    continue

                # we will receive a message containing "Please try again in <x>s."
                # x will be some decimal number of seconds (e.g., 12.132)
                try:
                    wait_time = float(re.search('Please try again in (\\d+\\.?\\d*)s', trial_response.json()['error']['message']).group(1))
                    print(f"Rate limit hit. Waiting for {wait_time} seconds.")
                    time.sleep(wait_time)
                    continue
                except KeyError as e:
                    # if we can't parse the wait time, then raise the error
                    print(trial_response.json())
                    raise e

            # make debugging easy
            print(trial_response)
            print(trial_response.json())
            raise e

        # Make sure the vision model response is valid.
        trial_parse_prompt = parse_prompt + "\n" + trial_response
        parse_payload["messages"][0]["content"][0]["text"] = trial_parse_prompt  # update the payload
        answer = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=parse_payload,
        )
        answer = answer.json()["choices"][0]["message"]["content"]
        i += 1
        time.sleep(2)  # not too many requests in a short period of time
    return answer, trial_response


def run_trial_azure(
    img_path: Union[str, List[str]],
    api_info: Dict[str, str],
    task_payload: Dict,
    parse_payload: Dict,
    parse_prompt: str,
    n_attempts: int = 5,
    max_tokens: int = 200,
):
    """
    Run a trial of the serial search task.

    Parameters:
    img_path (str): The path to the image for the trial.
    api_info(Dict[str, str]): The API information.
    task_payload (Dict): The payload for the vision model request.
    parse_payload (Dict): The payload for the parsing model request.
    parse_prompt (str): The prompt for the parsing model.
    n_attempts (int): The number of attempts to make. Default is 5.
    max_tokens (int): The maximum number of tokens for the API request. Default is 200.

    Returns:
    str: The response and the parsed response from the trial.
    """
    # Update the max number of tokens.
    task_payload["max_tokens"] = max_tokens

    # Encode the image and update the task payload.
    if isinstance(img_path, (list, tuple)):
        image = encode_image(img_path[0])
        task_payload["messages"][0]["content"][1]["image_url"]["url"] = f"data:image/jpeg;base64,{image}"

        for i, path in enumerate(img_path[1:]):
            image = encode_image(path)
            task_payload["messages"][0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}"
                }
            })
    else:
        image = encode_image(img_path)
        task_payload["messages"][0]["content"][1]["image_url"]["url"] = f"data:image/jpeg;base64,{image}"

    # Until the model provides a valid response, keep trying.
    answer = "-1"  # -1 if model failed to provide a valid response.
    i = 0  # Counter to keep track of the number of attempts. Limit to n_attempts.
    trial_response = None
    while answer == "-1" and i < n_attempts:
        # Get the vision model response.
        trial_response = requests.post(
            "https://gpt4-ilia-2024-switzerland-north.openai.azure.com/openai/deployments/gpt-4-vision-preview/chat/completions?api-version=2023-12-01-preview",
            headers=get_header(api_info, model='gpt-4-vision-azure'),
            json=task_payload,
        )

        # check for easily-avoidable errors
        if 'error' in trial_response.json():
            if 'API key' in trial_response.json()['error']['message']:
                raise ValueError("API key is invalid.")
            if trial_response.json()['error']['code'] == '404':
                raise Exception(trial_response.json())
        try:
            trial_response = trial_response.json()["choices"][0]["message"]["content"]
        except KeyError as e:
            # If we've hit a rate limit, then wait and try again.
            if 'error' in trial_response.json():
                # possibly encountered an error
                if 'Please try again in' not in trial_response.json()['error']['message']:
                    n_attempts += 1
                    continue

                # we will receive a message containing "Please try again in <x>s."
                # x will be some decimal number of seconds (e.g., 12.132)
                try:
                    wait_time = float(re.search('Please try again in (\\d+\\.?\\d*)s', trial_response.json()['error']['message']).group(1))
                    print(f"Rate limit hit. Waiting for {wait_time} seconds.")
                    time.sleep(wait_time)
                    continue
                except KeyError as e:
                    # if we can't parse the wait time, then raise the error
                    print(trial_response.json())
                    raise e

            # make debugging easy
            print(trial_response)
            print(trial_response.json())
            raise e

        # Make sure the vision model response is valid.
        trial_parse_prompt = parse_prompt + "\n" + trial_response
        parse_payload["messages"][0]["content"][0]["text"] = trial_parse_prompt  # update the payload
        answer = requests.post(
            "https://gpt4-ilia-2024-switzerland-north.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2023-12-01-preview",
            headers=get_header(api_info, model='gpt-4-vision-azure'),
            json=parse_payload,
        )
        answer = answer.json()["choices"][0]["message"]["content"]
        i += 1
        print('waiting for 45 seconds')
        time.sleep(45)
    return answer, trial_response


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
    argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run serial search trials.")
    parser.add_argument("--task", type=str, required=True, choices=["search", "popout", "counting", "describe", "rmts"], help="Which task to run.")
    parser.add_argument("--api_key", type=str, default="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", help="OpenAI API key.")
    parser.add_argument("--task_payload", type=str, default="payloads/gpt4v_single_image.json", help="The path to the task payload JSON file.")
    parser.add_argument("--parse_payload", type=str, default="payloads/gpt4_parse.json", help="The prompt for parsing the response.")
    return parser.parse_args()


def main():
    # Parse command line arguments.
    args = parse_args()

    # Load the relevant payloads and prompts.
    task_payload = json.load(open(args.task_payload, "r"))
    parse_payload = json.load(open(args.parse_payload, "r"))
    parse_prompt = Path(f"prompts/parse_{args.task}.txt").read_text()
    task_prompt = Path(f"prompts/run_{args.task}.txt").read_text()
    task_payload["messages"][0]["content"][0]["text"] = task_prompt

    # Open the results CSV file.
    results_df = pd.read_csv(f"output/{args.task}_results.csv", dtype={"path": str, "response": str, "answer": str})

    # OpenAI API Key and header.
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    # Run all the trials.
    for i, trial in tqdm(results_df.iterrows()):
        # Only run the trial if it hasn't been run before.
        if len(trial.response) != 0:
            try:
                answer, trial_response = run_trial(trial.path, headers, task_payload, parse_payload, parse_prompt)
                results_df.loc[i, "response"] = trial_response
                results_df.loc[i, "answer"] = answer
            except Exception as e:
                print(f"Failed on trial {i} with error: {e}")
                break  # Stop the loop if there is an error and save the progress.

    # Save the results.
    results_df.to_csv(f"output/{args.task}_results.csv", index=False)


if __name__ == "__main__":
    main()
