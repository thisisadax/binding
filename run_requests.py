import argparse
from glob import glob
import json
import requests
import time
from tqdm import tqdm
from typing import Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
import openai
from openai import OpenAI


from utils import encode_image

# NOTE (declan): Is it worth it to add exponential backoff with tenacity here?
def run_search_trial(img_path: str, 
					 headers: Dict[str, str], 
					 task_payload: Dict, 
					 parse_payload: Dict,
					 parse_prompt: str,
					 n_attempts: int = 5, 
					 max_tokens: int = 200) -> str:
	"""
	Run a trial of the serial search task.

	Parameters:
	img_path (str): The path to the image for the trial.
	headers (Dict[str, str]): The headers for the API request.
	task_payload (Dict): The payload for the vision model request.
	parse_payload (Dict): The payload for the parsing model request.
	parse_prompt (str): The prompt for the parsing model.
	n_attempts (int): The number of attempts to make. Default is 5.
	max_tokens (int): The maximum number of tokens for the API request. Default is 200.

	Returns:
	str: The response and the parsed response from the trial.
	"""
	# Encode the image and update the task payload.
	image = encode_image(img_path)
	task_payload['messages'][0]['content'][1]['image_url']['url'] = f'data:image/jpeg;base64,{image}'
	
	# Until the model provides a valid response, keep trying.
	answer = '-1' # -1 if model failed to provide a valid response.
	i = 0 # Counter to keep track of the number of attempts. Limit to n_attempts.
	while answer == '-1' and i < n_attempts:
		try:
			# Get the vision model response.
			trial_response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=task_payload)
			trial_response = trial_response.json()['choices'][0]['message']['content']

			# Make sure the vision model response is valid.
			trial_parse_prompt = parse_prompt + '\n' + trial_response
			parse_payload['messages'][0]['content'][0]['text'] = trial_parse_prompt # update the payload
			answer = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=parse_payload)
			answer = answer.json()['choices'][0]['message']['content']
		except KeyError as e:
			print('    Failed on attempt: ', i, ' with error: ', e)
			i += 1
	return answer, trial_response


def parse_args() -> argparse.Namespace:
	"""
	Parse command line arguments.

	Returns:
	argparse.Namespace: The parsed command line arguments.
	"""
	parser = argparse.ArgumentParser(description='Run serial search trials.')
	parser.add_argument('--task', type=str, required=True, choices=['search', 'popout', 'counting', 'describe', 'rmts'], help='Which task to run.')
	parser.add_argument('--api_key', type=str, default='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', help='OpenAI API key.')
	parser.add_argument('--task_payload', type=str, default='payloads/gpt4v_single_image.json', help='The path to the task payload JSON file.')
	parser.add_argument('--parse_payload', type=str, default='payloads/gpt4_parse.json', help='The prompt for parsing the response.')
	return parser.parse_args()


def main():
	# Parse command line arguments.
	args = parse_args()

	# Load the relevant payloads and prompts.
	task_payload = json.load(open(args.task_payload, 'r'))
	parse_payload = json.load(open(args.parse_payload, 'r'))
	parse_prompt = Path(f'prompts/parse_{args.task}.txt').read_text()
	task_prompt = Path(f'prompts/run_{args.task}.txt').read_text()
	task_payload['messages'][0]['content'][0]['text'] = task_prompt

	# Open the results CSV file.
	results_df = pd.read_csv(f'output/{args.task}_results.csv', dtype={'path': str, 'response': str, 'answer': str})

	# OpenAI API Key and header.
	headers = {
		'Content-Type': 'application/json',
		'Authorization': f'Bearer {args.api_key}'
	}
	# Run all the trials.
	for i, trial in tqdm(results_df.iterrows()):
		if type(trial.response)!=str:
			answer, trial_response = run_search_trial(trial.path, headers, task_payload, parse_payload, parse_prompt)
			results_df.loc[i, 'response'] = trial_response
			results_df.loc[i, 'answer'] = answer

	# Save the results.
	results_df.to_csv(f'output/{args.task}_results.csv', index=False)

if __name__ == '__main__':
	main()