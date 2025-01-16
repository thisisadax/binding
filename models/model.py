import json
import os
from utils import get_header
import time
import numpy as np
import pandas as pd
from tasks.task import Task
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import warnings
from typing import Dict
warnings.filterwarnings('ignore')


class Model():
    """
    Base Model class that other models will inherit from.
    """
    def __init__(
            self,
            task: Task
    ):
        self.task = task

    def run(self):
        print('Need to specify a particular model class.')
        raise NotImplementedError
    
    def save_results(self, results_file: str=None):
        """
        Save the task results to a CSV file.
        """
        if results_file:
            self.task.results_df.to_csv(results_file, index=False)
        else:
            filename = f'results_{time.time()}.csv'
            self.task.results_df.to_csv(filename, index=False)

class ParseModel(Model):
    """
    Model class responsible for parsing responses from an LLM.
    """
    def __init__(
            self,
            model_name: str,
            payload_path: str,
            api_file: str,
            sleep: int = 0,
            prompt_path: str = None
    ):
        ### [CHECK] Shouldn't we add this: super().__init__(task=None) ?
        self.model_name = model_name
        ### [CHECK] Wanna keep the name good_models?
        good_models = ['gpt4']
        assert self.model_name in good_models, (
            f"Model name must be one of {good_models}, not {self.model_name}"
        )

        with open(payload_path) as f:
            self.payload = json.load(f)
        with open(api_file, 'r') as f:
            self.api_metadata = json.load(f)

        self.prompt = open(prompt_path, 'r').read() if prompt_path else ""
        self.header = get_header(self.api_metadata, model=self.model_name)
        self.api_metadata = self.api_metadata[self.model_name]
        self.sleep = sleep


class APIModel(Model):
    """
    Model class for interacting with various APIs.
    """
    def __init__(
            self,
            task: Task,
            model_name: str,
            payload_path: str,
            api_file: str,
            sleep: int = 0,
            shuffle: bool = False,
            n_trials: int = None,
            parse_model: ParseModel = None
    ):
        self.model_name = model_name
        good_models = ['gpt4v', 'gpt4o', 'claude-sonnet', 'claude-opus', 'gemini-ultra', 'stable-diffusion', 'dalle', 'parti', 'muse']
        assert self.model_name in good_models, f'Model name must be one of {str(good_models)}, not {self.model_name}'
        super().__init__(task)
        self.payload = json.load(open(payload_path))
        self.api_metadata = json.load(open(api_file, 'r'))
        self.header = get_header(self.api_metadata, model=self.model_name)
        self.api_metadata = self.api_metadata[self.model_name]
        self.sleep = sleep
        self.results_file = self.task.results_path
        self.shuffle = shuffle
        self.n_trials = n_trials
        # Define the parse model, if necessary.
        self.parse_model = parse_model
        if self.parse_model:
            self.task.results_df['answer'] = ''
            print(f'Parse model: {self.parse_model.model_name}')
        # Shuffle and subsample the task dataset, if necessary
        if self.shuffle:
            if self.n_trials and self.n_trials > 0:
                self.task.results_df = self.task.results_df.sample(n=self.n_trials)
            else:
                self.task.results_df = self.task.results_df.sample(frac=1)

    #@retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(10))
    def run(self):
        
        if self.task.num_remaining_trials() == 0:
            return
        
        p_bar = tqdm(total=self.task.num_remaining_trials())
        
        for i, trial in self.task.results_df.iterrows():
            if type(trial.response) != str or trial.response == '0.0':
                trial_payload = self.payload.copy()
                task_payload = self.build_vlm_payload(trial, trial_payload)
                response = self.run_trial(self.header, self.api_metadata, task_payload)
                self.task.results_df.loc[i, 'response'] = response

                # Parse the response, if necessary.
                if self.parse_model:
                    self.task.results_df.loc[i, 'answer'] = self.parse_model.parse_response(response)

                p_bar.update()
                time.sleep(self.sleep)
                
            if i % 1 == 0:
                self.save_results(self.results_file)
                
        self.save_results(self.results_file)


class T2IModel(APIModel):
    """
    Text-to-Image Model class.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.img_path = os.path.join(self.task.output_dir, 't2i', self.task.task_name, self.model_name)
        os.makedirs(self.img_path, exist_ok=True)

    #@retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(10))
    def run(self):
        p_bar = tqdm(total=self.task.num_remaining_trials())
        for i, trial in self.task.results_df.iterrows():
            if not bool(trial.completed):
                trial_payload = self.payload.copy()
                task_payload = self.build_payload(trial, trial_payload)
                image, revised_prompt = self.run_trial(self.header, self.api_metadata, task_payload)
                self.task.results_df.loc[i, 'revised_prompt'] = revised_prompt
                self.task.results_df.loc[i, 'completed'] = True
                path = os.path.join(self.img_path, trial.path)
                self.task.results_df.loc[i, 'path'] = path
                image.save(path)
                p_bar.update()
                time.sleep(self.sleep)
            if i % 1 == 0:
                self.save_results(self.results_file)
        self.save_results(self.results_file)

class LocalLanguageModel(Model):
    """
    Local Language Model class for huggingface models.
    """
    def __init__(
        self,
        task: Task = None,
        max_parse_tokens: int = 256,
        prompt_format: str = None,
        weights_path: str = None,
        probe_layers: Dict = None
    ):
        super().__init__(task)
        if task:
            self.results_file = self.task.results_path
        self.max_parse_tokens = max_parse_tokens
        self.prompt_format = prompt_format
        self.weights_path = weights_path
        self.probe_layers = probe_layers
        self.prompt = ''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.activations = {}
        self.llm = AutoModelForCausalLM.from_pretrained(self.weights_path, device_map='auto', torch_dtype='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(self.weights_path, use_fast=True)
        self.llm.eval()
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.target_column = 'response'

    def run_batch(self, batch):
        prompts = [p.format(text_to_parse=t) for p, t in zip([self.prompt]*len(batch), batch[self.target_column])]
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding='longest',
            truncation=True,
        )
        inputs.to(self.device)
        with torch.no_grad():
            outputs = self.llm.generate(**inputs, max_new_tokens=self.max_parse_tokens)
        outputs = [output[inputs.input_ids.shape[1]:] for output in outputs]
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i, o in zip(prompts, decoded_outputs):
            print(f'Prompt: {i}\nResponse: {o}\n')
        batch['answer'] = decoded_outputs
        return batch

class LocalVLModel(Model):
    """
    Local Vision-Language Model class for processing multimodal tasks.
    """
    def __init__(
        self,
        task: Task,
        max_tokens: int = 512,
        batch_size: int = 32,
        weights_path: str = None,
        shuffle: bool = False,
        n_trials: int = None,
        model_name: str = None
    ):
        super().__init__(task)
        self.results_file = self.task.results_path
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.weights_path = weights_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_trials = n_trials
        self.shuffle = shuffle
        self.model_name = model_name

        # shuffle and subsample the task dataset, if necessary
        if self.shuffle:
            if self.n_trials and self.n_trials > 0:
                self.task.results_df = self.task.results_df.sample(n=self.n_trials)
            else:
                self.task.results_df = self.task.results_df.sample(frac=1)

    def run(self):
        results = []
        batches = np.array_split(self.task.results_df, np.ceil(len(self.task.results_df)/self.batch_size))
        for i, batch in tqdm(enumerate(batches)): 
            batch = self.run_batch(batch)
            results.append(batch)
            if i % 10 == 0:
                self.task.results_df = pd.concat(batches[i:]+results)
                self.save_results(self.results_file)