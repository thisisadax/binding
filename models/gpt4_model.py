import ast
import requests
from utils import encode_image
from models.model import APIModel, ParseModel


class GPT4Model(APIModel):

    def __init__(self, max_tokens: int = 512, **kwargs):
        self.max_tokens = max_tokens
        super().__init__(**kwargs)
        self.payload['max_tokens'] = self.max_tokens

    def build_vlm_payload(self, trial_metadata, task_payload):
        """
        Parameters:
            trial_metadata (dict): The metadata for the task.
            task_payload (str): The task payload.

        Returns:
            str: The parsed task prompt.
        """
        task_payload['messages'][0]['content'][0]['text'] = self.task.prompt
        task_payload['messages'][0]['content'] = [task_payload['messages'][0]['content'][0]]
        img_path = trial_metadata['path']
        images = [encode_image(img_path)]
        # Add the images to the payload
        image_payload = [{'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image}'}} for image in images]
        task_payload['messages'][0]['content'] += image_payload
        return task_payload

    def run_trial(self, header, api_metadata, task_payload):
        """
        Parameters:
            header (dict): The header for the API request.
            api_metadata (dict): The metadata for the API.
            task_payload (dict): The payload for the task.

        Returns:
            str: The response.
        """
        
        # Until the model provides a valid response, keep trying.
        trial_response = requests.post(
            api_metadata['endpoint'],
            headers=header,
            json=task_payload,
            timeout=300
        )

        # Check for easily-avoidable errors.
        if 'error' in trial_response.json():
            error = str(trial_response.json()['error']['message'])
            raise ValueError('Returned error: \n' + error)
        response = trial_response.json()['choices'][0]['message']['content']
        return response
    

class GPT4RMTSModel(GPT4Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_vlm_payload(self, trial_metadata, task_payload):
        """
        Parameters:
            trial_metadata (dict): The metadata for the task.
            task_payload (str): The task payload.
            task (str): The task name.

        Returns:
            str: The parsed task prompt.
        """
        task_payload['messages'][0]['content'][0]['text'] = self.task.prompt
        task_payload['messages'][0]['content'] = [task_payload['messages'][0]['content'][0]]
        vals = {k: v for k, v in trial_metadata.items() if f'{{{k}}}' in task_payload['messages'][0]['content'][0]['text']}
        if self.task.subtask != 'feature2':
            task_payload['messages'][0]['content'][0]['text'] = task_payload['messages'][0]['content'][0]['text'].format(**vals)
        else:
            task_payload['messages'][0]['content'][0]['text'] = task_payload['messages'][0]['content'][0]['text']
        if self.task.condition == 'decomposed':
            img_path = ast.literal_eval(trial_metadata['decomposed_paths'])
            images = [encode_image(path) for path in img_path]
        else:  # unified
            img_path = trial_metadata['unified_path']
            images = [encode_image(img_path)]

        # Add the images to the payload
        image_payload = [{'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image}'}} for image in images]
        task_payload['messages'][0]['content'] += image_payload
        return task_payload


class GPT4ParseModel(ParseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parse_response(self, response):
        trial_parse_prompt = self.prompt + '\n' + response
        self.payload['messages'][0]['content'][0]['text'] = trial_parse_prompt  # update the payload
        answer = requests.post(self.api_metadata['endpoint'], headers=self.header, json=self.payload, timeout=240)
        answer = answer.json()['choices'][0]['message']['content']

        # If the response is invalid raise an error.
        if 'error' in answer:
            print('failed parsing request')
            raise ValueError('Returned error: \n' + answer['error']['message'])
        elif answer == '-1':
            print(f'bad VLM response: {response}')
            raise ValueError(f'Invalid response: {response}')
        return answer