import os
import requests
from models.model import T2IModel
from PIL import Image
from io import BytesIO


class DALLEModel(T2IModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_payload(self, trial_metadata, task_payload):
        """
        Parameters:
            trial_metadata (dict): The metadata for the task.
            task_payload (str): The task payload.

        Returns:
            str: The parsed task prompt.
        """
        task_payload['prompt'] = trial_metadata.prompt
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
        response = requests.post(api_metadata['endpoint'], headers=header, json=task_payload, timeout=45)

        if 'error' in response.json():
            error = str(response.json()['error']['message'])
            raise ValueError('Returned error: \n' + error)
        
        # Extract the responses from the vision model and parse them with the parsing model.
        image_url = response.json()['data'][0]['url']
        # Get the refined prompt
        if 'revised_prompt' in response.json()['data'][0]:
            revised_prompt = response.json()['data'][0]['revised_prompt']
        else:
            revised_prompt = task_payload['prompt']
        image = requests.get(image_url).content
        image = Image.open(BytesIO(image))
        return image, revised_prompt