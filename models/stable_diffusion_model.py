import os
import requests
from models.model import T2IModel
from PIL import Image
from io import BytesIO
import base64


class StableDiffusionModel(T2IModel):

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
        # Return the requested image.
        response = requests.post(api_metadata['endpoint'], headers=header, files={'none': ''}, data=task_payload)
        image_base64_str = response.json()['image']
        image_bytes = base64.b64decode(image_base64_str)
        image = Image.open(BytesIO(image_bytes))
        return image, task_payload['prompt'] # SD doesn't change the prompt