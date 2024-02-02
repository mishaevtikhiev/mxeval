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
    def __init__(self, model_checkpoint: str = None, model_name = "smallcloudai/Refact-1_6B-fim", gpu_id=2, method_dict_location:str=None):
        super().__init__(model_checkpoint, model_name, gpu_id, method_dict_location)
        self.raw_method_list = None

    def _init_model(self, model_checkpoint: str = None, model_name: str = "smallcloudai/Refact-1_6B-fim", gpu_id: int = 2):
        self.model_name = 'Refact'
        if model_name is not None:
            self.model_name = model_name.split('/')[-1]
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if torch.cuda.is_available():
                self._device = "cuda:" + str(gpu_id)
            else:
                self._device = "cpu"
            if model_checkpoint is not None:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(self._device)
                checkpoint = torch.load(model_checkpoint)
                self.model.load_state_dict(checkpoint)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(self._device)

    def method_filter(self, answer: str):
        answer_start = answer.find('{\n')
        answer_end = answer.find('\n}\n\n', answer_start + 1)
        return answer[answer_start + 1:answer_end + 2]
        # This one is for Refact. We're looking for final curly bracket at the end of method, after method body

    def _data_unwrapper(self, problem_list: dict[str, dict]):
        for key in problem_list.keys():
            prompt = problem_list[key]['prompt']
            yield self.tokenizer.encode(prompt, return_tensors="pt").to(self._device)

    def model_generate(self, problem_list, top_k = 1, max_length = 500):
        method_list = list()
        raw_output = list()
        for problem in tqdm(problem_list):
            sample = self.model.generate(problem, temperature=0.2, max_length=max_length)
            if len(sample[0]) < 3: # say no to <endoftext> instead of answer
                i = 0
                while len(sample[0]) < 3 & i < 5:
                    sample = self.model.generate(problem, temperature=0.2, max_length=max_length)
                    i += 1
            answer = self.tokenizer.decode(sample[0])
            method_list.append(self.method_filter(answer))
            raw_output.append(answer)
        self.raw_method_list = raw_output
        return method_list

    def generate(self, problem_name: str = 'multi-humaneval', top_k = 1, max_length = 500):
        super().generate(problem_name, top_k, max_length)

    def model_process(self, problem_name: str = 'multi-humaneval'):
        output_file = f'output_{problem_name}_{self.model_name}.jsonl'
        with jsonlines.open(output_file, mode='w') as writer:
            for key, value in self.method_dict.items():
                # answer_start = value.find('*/')
                # body_start = value.find('\n', answer_start + 3)
                # answer = value[body_start:]
                generated_sample = {"task_id": key, "completion": value, "language": "kotlin"}
                writer.write(generated_sample)
        return output_file

    def evaluate(self, problem_name: str = 'humaneval', model_outputs_jsonl: str = 'output_multi-humaneval_Refact-1_6B-fim.jsonl',
                     top_k=1, n_workers=8, timeout=15.0):
        return super().evaluate(problem_name, model_outputs_jsonl, top_k, n_workers, timeout)

    def multi_evaluate(self, iterations: int = 1, problem_name: str = 'humaneval', top_k = 1, max_length = 500, model_outputs_jsonl: str = 'output_multi-humaneval_Refact-1_6B-fim.jsonl', n_workers=8, timeout=15.0):
        for i in range(iterations):
            if problem_name == 'humaneval':
                long_name = 'multi-humaneval'
            else:
                long_name = 'mbkp'
            self.generate(long_name, top_k, max_length)
            self.model_process(long_name)
            print(f"Iteration {i+1} out of {iterations+1}")
            print(self.evaluate(problem_name, model_outputs_jsonl, top_k, n_workers, timeout))
        mean_scores = dict()
        for (key, value) in self.scores.items():
            mean_scores[key] = sum(self.scores[key]) / len(self.scores[key])
        return mean_scores
