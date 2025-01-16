import ast
import requests
from utils import encode_image
from models.model import APIModel


class ClaudeModel(APIModel):

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
        image_payload = [{'type': 'image', 'source': 
                            {'type': 'base64', 'media_type': 'image/png', 'data': image}
                         } for image in images]
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
            timeout=240
        )

        # Check for easily-avoidable errors.
        if 'error' in trial_response.json():
            raise ValueError('Returned error: \n' + trial_response.json()['error']['message'])
        response = trial_response.json()['content'][0]['text']
        return response
    

class ClaudeRMTSModel(ClaudeModel):

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
        image_payload = [{'type': 'image', 'source': {
            'type': 'base64', 'media_type': 'image/png', 'data': image}
        } for image in images]
        task_payload['messages'][0]['content'] += image_payload
        return task_payload