from models.model import LocalVLModel
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import subprocess as sp
import torch


class LlavaModel(LocalVLModel):
        
    def __init__(self, prompt_format, **kwargs):
        super().__init__(**kwargs)
        self.task.prompt = prompt_format.format(prompt=self.task.prompt)
        self.processor = LlavaNextProcessor.from_pretrained(self.weights_path)
        self.processor.tokenizer.padding_side = 'left'
        self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
        self.model = LlavaNextForConditionalGeneration.from_pretrained(self.weights_path, device_map='auto')
        self.model.eval()
        
    def run_batch(self, batch):
        image_paths = batch.path.values
        inputs_batched = self.processor(
            [self.task.prompt] * len(image_paths),
            images=[Image.open(image_path) for image_path in image_paths],
            return_tensors='pt',
            padding=True)
        inputs_batched.to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs_batched, max_new_tokens=self.max_tokens, pad_token_id=self.processor.tokenizer.eos_token_id)
        outputs = [output[inputs_batched.input_ids.shape[1]:] for output in outputs]
        generated_texts = self.processor.batch_decode(outputs, skip_special_tokens=True)
        batch['response'] = generated_texts
        return batch

    def get_gpu_memory(self):
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        return memory_free_values