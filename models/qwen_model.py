from models.model import LocalVLModel
from lmdeploy import pipeline
from PIL import Image
import subprocess as sp

class QwenModel(LocalVLModel):
        
    def __init__(self, prompt_format, **kwargs):
        super().__init__(**kwargs)
        self.pipe = pipeline(model_path=self.weights_path)
        
    def run_batch(self, batch):
        image_paths = batch.path.values
        prompts = [(self.task.prompt, Image.open(image_path)) for image_path in image_paths]
        generated_texts = self.pipe(prompts)
        batch['response'] = generated_texts
        return batch

    def get_gpu_memory(self):
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        return memory_free_values