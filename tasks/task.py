import os
from typing import List
from pathlib import Path
import pandas as pd

class Task:
    def __init__(self, 
                 task_name=None, 
                 task_variant=None, 
                 model_name=None, 
                 root_dir=None, 
                 output_dir=None, 
                 data_dir=None,
                 metadata_file=None,
                 prompt_path = None):
        self.task_name = task_name
        self.task_variant = task_variant
        self.model_name = model_name
        self.run_id = self.model_name + '_' + self.task_name + '_' + self.task_variant
        self.task_id = self.task_name + '_' + self.task_variant
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.metadata_file = metadata_file
        self.prompt = Path(prompt_path).read_text()
        if self.task_name != 'rmts':
            outpath = os.path.join(self.output_dir, 'vlm', self.task_variant, self.task_name)
            self.results_path = os.path.join(outpath, f'{self.model_name}.csv')
        else:
            outpath = os.path.join(self.output_dir, 'vlm', self.task_variant, self.task_name, self.condition, self.subtask)
            self.results_path = os.path.join(outpath, f'{self.model_name}.csv')
        os.makedirs(outpath, exist_ok=True)
        task_path = os.path.join(self.data_dir, 'vlm', self.task_variant, self.task_name, self.metadata_file)
        if os.path.exists(self.results_path):
            print(f'Loading task metadata from {self.results_path}...')
            self.results_df = pd.read_csv(self.results_path)
        elif os.path.exists(task_path):
            print(f'Loading task metadata from {task_path}...')
            self.results_df = pd.read_csv(task_path)
        else:
            print(f'No dataset found at {self.results_path} or {task_path}.')
            print('Generating full dataset...')
            if self.task_variant == '2D':
                self.results_df = self.generate_full_dataset()
                print(f'Saving metadata to {task_path}...')
                self.results_df.to_csv(task_path, index=False)
            else:
                raise AttributeError('3D dataset can not be generated. Use gen_blender.py to create.')
            return None

    def generate_full_dataset(self):
        raise NotImplementedError

    def num_remaining_trials(self):
        return self.results_df['response'].isna().sum()


class T2ITask:
    def __init__(self, 
                 task_name=None, 
                 model_name=None, 
                 root_dir=None, 
                 output_dir=None, 
                 data_dir=None,
                 metadata_file=None,
                 prompt_path = None):
        self.task_name = task_name
        self.model_name = model_name
        self.run_id = self.model_name + '_' + self.task_name
        self.task_id = self.task_name 
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.metadata_file = metadata_file
        self.prompt = Path(prompt_path).read_text()
        outpath = os.path.join(self.output_dir, 't2i', self.task_name)
        self.results_path = os.path.join(outpath, f'{self.model_name}.csv')
        os.makedirs(outpath, exist_ok=True)
        task_path = os.path.join(self.data_dir, 't2i', self.metadata_file)
        if os.path.exists(self.results_path):
            print(f'Loading task metadata from {self.results_path}...')
            self.results_df = pd.read_csv(self.results_path)
        elif os.path.exists(task_path):
            print(f'Loading task metadata from {task_path}...')
            self.results_df = pd.read_csv(task_path)
        else:
            print('Generating full dataset...')
            self.results_df = self.generate_full_dataset()
            self.results_df.to_csv(task_path, index=False)


    def generate_full_dataset(self):
        raise NotImplementedError

    def num_remaining_trials(self):
        pass