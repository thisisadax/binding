import argparse
import base64
from glob import glob
import re
import requests
import time
import tqdm
from typing import Dict, Any

import numpy as np
import pandas as pd
import openai
from openai import OpenAI


# TODO (declan): Update prompt to most recent version.
# TODO (declan): Add all prompts to unified prompt file?
task_prompt = '''
The following image depicts a classic serial search task from cognitive psychology.
The image contains several objects with two different shape types (e.g. hashtags and triangles) and two different colors (e.g. red and blue). 
Your task is to determine if all objects of the same shape type also have the same color. 
If all objects of the same shape have the same color, respond with "0", and if some objects of the same shape type have different colors, respond with "1".
For example, if all of the hashtags are red and all of the triangles are blue, then all of the shapes of the same type have the same color, so you should respond with “0”. 
On the other hand, if some of the hashtags are red and some of the triangles are blue, you should respond with “1”.
The object types aren't limited to hashtags and triangles, and the colors aren't limited to red and blue.
Start your answer by describing the objects that you see in the image, and then provide your answer.
'''

# TODO (declan): Update prompt to most recent version.
parse_prompt = '''
You are responsible for evaluating the responses of participants in a simple serial search task.
Your job is to evaluate whether the participants responded to the question appropriately, and parse their answer into a simple binary response.
If the participants responded with "0", then you should respond with "0" and if the participants responded with "1", then you should respond with "1".
However, if the participant attempted to answer, but did not indicate their choice with a simple binary response, then you should respond with "-1".
Moreover, if the participants omitted their response, or if their response was not a simple binary response, then you should respond with "-1".
Include absolutely no additional justification, text, or delimiting characters in your response other than 0, 1, or -1.

Here are some examples of responses with their corresponding evaluations:
Response 1: The image contains two types of shapes: two blue summation symbols (Σ) and two red square brackets. Both shape types are consistent in color; all summation symbols are blue and all square brackets are red. Therefore, all objects of the same shape type also have the same color. The answer is "0".
Answer 1: 0

Response 2: In the image, there are objects that appear to be snowmen and triangles. The snowmen are all the same color, red. However, the triangles come in two colors: most are green, but there is one triangle that is red. Since some of the triangles have different colors, the answer is "1".
Answer 2: 1

Response 3: I'm sorry, but I can't assist with this request.
Answer 3: -1

Response 4: I'm sorry, but I am unable to process requests that require the analysis of text within images. If you can provide the details in another format, I would be more than happy to help you.
Answer 4: -1
'''


def parse_response(trial_response: str, client: openai.api_resources.completion.Completion, prompt: str) -> str:
	"""
	Evaluate whether the model responses are valid or not, and parse the response integer.

	Parameters:
	trial_response (str): The response from the trial.
	client (openai.api_resources.completion.Completion): The OpenAI client.
	prompt (str): The prompt for the trial.

	Returns:
	str: The parsed response.
	"""
	prompt = prompt + 'Here is the participant response: ' + trial_response
	messages = [{'role': 'system', 'content': 'You are a helpful assistant designed to help parse participant responses in a psychology study.'},
				{'role': 'user', 'content': f'{prompt}'}]
	response = client.chat.completions.create(model='gpt-4-1106-preview', messages=messages)
	return response.choices[0].message.content


# TODO (declan): Adapt this function to be more general so that we can use it across all tasks.
def run_search_trial(img_path: str, 
					 headers: Dict[str, str], 
					 client: openai.api_resources.completion.Completion, 
					 task_prompt: str, 
					 parse_prompt: str, 
					 n_attempts: int = 5, 
					 max_tokens: int = 200) -> str:
	"""
	Run a trial of the serial search task.

	Parameters:
	img_path (str): The path to the image for the trial.
	headers (Dict[str, str]): The headers for the API request.
	client (openai.api_resources.completion.Completion): The OpenAI client.
	task_prompt (str): The task prompt for the trial.
	parse_prompt (str): The parse prompt for the trial.
	n_attempts (int): The number of attempts to make. Default is 5.
	max_tokens (int): The maximum number of tokens for the API request. Default is 200.

	Returns:
	str: The response from the trial.
	"""
	def encode_image(image_path):
		with open(image_path, "rb") as image_file:
			return base64.b64encode(image_file.read()).decode('utf-8')
	image = encode_image(img_path)
	payload = {
		"model": "gpt-4-vision-preview",
		"messages": [
			{"role": "user",
			"content": [
				{"type": "text",
				"text": task_prompt},
				{"type": "image_url",
				"image_url": {
					"url": f"data:image/jpeg;base64,{image}"}}]}],
			"max_tokens": max_tokens}
	# Until the model provides a valid response, keep trying.
	answer = '-1' # This is if the model failed to provide a valid response.
	i = 0 # Counter to keep track of the number of attempts. Limit to n_attempts.
	while answer == '-1' and i < n_attempts:
		try:
			trial_response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload)
			trial_response = trial_response.json()['choices'][0]['message']['content']
			answer = parse_response(trial_response, client, parse_prompt)
		except KeyError as e:
			print('    Failed on attempt: ', i)
	return answer, trial_response


def parse_args() -> argparse.Namespace:
	"""
	Parse command line arguments.

	Returns:
	argparse.Namespace: The parsed command line arguments.
	"""
	parser = argparse.ArgumentParser(description='Run serial search trials.')
	parser.add_argument('--api_key', type=str, default='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', help='OpenAI API key.')
	return parser.parse_args()


def main():
	# Parse command line arguments.
	args = parse_args()
	 
	# load incongruent and congruent trials from serial_trials.
	congruent_trials = glob('./data/serial_search/congruent*.png')
	incongruent_trials = glob('./data/serial_search/incongruent*.png')
	n_trials =  len(incongruent_trials) + len(congruent_trials)
	results_df = pd.DataFrame(np.zeros([n_trials, 5]), columns=['path', 'incongruent', 'n_shapes', 'response', 'answer'])
	results_df['path'] = incongruent_trials + congruent_trials
	results_df['incongruent'] = [True if 'incongruent' in path else False for path in results_df['path']]
	results_df['n_shapes'] = [re.split(r'-|_', path)[2] for path in results_df['path']]
	results_df['response'] = results_df['response'].astype('object')
	results_df['answer'] = results_df['answer'].astype('object')

	# OpenAI API Key and header.
	headers = {
	'Content-Type': 'application/json',
	'Authorization': f'Bearer {args.api_key}'
	}
	client = OpenAI(api_key=args.api_key)

	# TODO (declan): Update to use tenacity exponential backoff.
	for i, trial in tqdm(results_df.iterrows()):
		if type(trial.response)!=str:
			time.sleep(5)
			answer, trial_response = run_search_trial(trial.path, headers, client, task_prompt, parse_prompt)
			results_df.loc[i, 'response'] = trial_response
			results_df.loc[i, 'answer'] = answer