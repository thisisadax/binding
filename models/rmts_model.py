import requests
from models.model import APIModel


class RMTSModel(APIModel):
    ### [CHECK] Should this be moved to the model file, since it's not Model-specific?
    def __init__(self, max_tokens: int = 512, **kwargs):
        self.max_tokens = max_tokens
        super().__init__(**kwargs)

    def run_trial(self, header, api_metadata, task_payload):
        """
        Parameters:
            header (dict): The header for the API request.
            api_metadata (dict): The metadata for the API.
            task_payload (dict): The payload for the task.
            parse_payload (dict): The payload for the parse.
            parse_prompt (str): The parse prompt.

        Returns:
            str: The answer.
            str: The response.
        """
        
        # Until the model provides a valid response, keep trying.
        trial_response = requests.post(
            api_metadata['vision_endpoint'],
            headers=header,
            json=task_payload,
            timeout=240
        )

        # Check for easily-avoidable errors
        if 'error' in trial_response.json():
            raise ValueError('Returned error: \n' + trial_response.json()['error']['message'])
        
        response = trial_response.json()['content'][0]['text']
        return response