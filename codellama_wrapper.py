from transformers import AutoTokenizer
import transformers
import wrapper
import torch
import mxeval
import json
import jsonlines
from tqdm import tqdm

from mxeval.data import get_data

class CodellamaEval(wrapper.KotlinLLMEval):

    def __init__(self, model_name = "codellama/CodeLlama-7b-hf", gpu_id=2, method_dict_location:str=None):
        super().__init__(model_name, gpu_id, method_dict_location)

    def _init_model(self, model_name: str = "codellama/CodeLlama-7b-hf", gpu_id: int = 0):
        self.model_name = 'Codellama'
        if model_name is not None:
            self.model_name = model_name.split('/')[-1]
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if torch.cuda.is_available():
                self._device = "cuda:" + str(gpu_id)
            else:
                self._device = "cpu"
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.float16,
                device=self._device,
            )

    def method_filter(self, answer: str):
        answer_start = answer.find('/**')
        answer_end = answer.find('/**', answer_start + 1)
        return answer[answer_start:answer_end]

    def unwrap_model_output(self, problem_list: dict[str, dict]):
        for key in problem_list.keys():
            yield problem_list[key]['prompt']

    def model_generate(self, problem_list, top_k = 1, max_length = 500):
        method_list = list()
        for sample in tqdm(self.pipeline(problem_list, do_sample=True, top_k=top_k, temperature=0.1, top_p=0.95,
                                         num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id,
                                         max_length=max_length)):
            answer = sample[0]['generated_text']
            method_list.append(self.method_filter(answer))
        return method_list

    def generate(self, problem_name: str = 'multi-humaneval', top_k = 1, max_length = 500):
        super().generate(problem_name, top_k, max_length)
       

    def model_process(self, problem_name: str = 'multi-humaneval'):
        output_file = f'output_{problem_name}_{self.model_name}.jsonl'
        with jsonlines.open(output_file, mode='w') as writer:
            for key, value in self.method_dict.items():
                answer_start = value.find('*/')
                body_start = value.find('\n', answer_start + 3)
                answer = value[body_start:]
                generated_sample = {"task_id": key, "completion": answer, "language": "kotlin"}
                writer.write(generated_sample)
        return  output_file


    def evaluate(self, problem_name: str  = 'humaneval', model_outputs_jsonl: str = 'outputmulti-humaneval.jsonl', top_k=1, n_workers=8, timeout=15.0):
        return super().evaluate(problem_name, model_outputs_jsonl, top_k, n_workers, timeout)





