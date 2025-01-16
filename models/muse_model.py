import base64
from models.model import T2IModel
from PIL import Image
from io import BytesIO
from google.auth.transport import requests as google_requests
from google.oauth2 import service_account


class MuseModel(T2IModel):

    def __init__(self, **kwargs):
        credentials = service_account.Credentials.from_service_account_file('google-cloud-account.json', scopes=['https://www.googleapis.com/auth/cloud-platform'])
        self.session = google_requests.AuthorizedSession(credentials)
        super().__init__(**kwargs)

    def build_payload(self, trial_metadata, task_payload):
        """
        Parameters:
            trial_metadata (dict): The metadata for the task.
            task_payload (str): The task payload.

        Returns:
            str: The parsed task prompt.
        """
        task_payload['instances'][0]['prompt'] = trial_metadata.prompt
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
        response = self.session.post(api_metadata['endpoint'], json=task_payload)
        image_response = response.json()['predictions'][0]['bytesBase64Encoded']
        image_bytes = base64.b64decode(image_response)
        image = Image.open(BytesIO(image_bytes))
        return image, task_payload['instances'][0]['prompt']