from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import wrapper
import torch
import mxeval
import json
import jsonlines
from tqdm import tqdm

from mxeval.data import get_data

class CausalLMEval(wrapper.KotlinLLMEval):
    def __init__(self, model_name = "smallcloudai/Refact-1_6B-fim", gpu_id=0, method_dict_location:str=None):
        super().__init__(model_name, gpu_id, method_dict_location)

    def _init_model(self, model_name: str = "smallcloudai/Refact-1_6B-fim", gpu_id: int = 0):
        self.model_name = 'Refact'
        if model_name is not None:
            self.model_name = model_name.split('/')[-1]
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if torch.cuda.is_available():
                self._device = "cuda:" + str(gpu_id)
            else:
                self._device = "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(self._device)

    def method_filter(self, answer: str):
        answer_start = answer.find('/**')
        answer_end = answer.find('/**', answer_start + 1)
        return answer[answer_start:answer_end]

    def _data_unwrapper(self, problem_list: dict[str, dict]):
        for key in problem_list.keys():
            prompt = problem_list[key]['prompt']
            yield self.tokenizer.encode(prompt, return_tensors="pt").to(self._device)

    def model_generate(self, problem_list, top_k = 1, max_length = 500):
        method_list = list()
        for sample in tqdm(self.pipeline(problem_list, do_sample=True, top_k=top_k, temperature=0.1, top_p=0.95,
                                         num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id,
                                         max_length=max_length)):
            answer = sample[0]['generated_text']
            method_list.append(self.method_filter(answer))
        return method_list
