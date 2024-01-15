from transformers import AutoTokenizer
import transformers
import torch
import mxeval
import json
import jsonlines
from tqdm import tqdm

from mxeval.data import get_data


class KotlinLLMEval:
    def __init__(self, model_name : str =None, gpu_id=0, method_dict:dict=None):
        self.method_dict = method_dict
        if model_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        # self.problems_humaneval = get_data(dataset='multi-humaneval', language='kotlin')
        # self.problems_mbkp = get_data(language='kotlin')

    def codellama_method_filter(self, answer: str):
        answer_start = answer.find('/**')
        answer_end = answer.find('/**', answer_start + 1)
        return answer[answer_start:answer_end]

    def data_unwrapper(self, problem_list: dict[str, dict]):
        for key in problem_list.keys():
            yield problem_list[key]['prompt']

    def codellama_generate(self, problem_name: str, top_k = 1):
        raw_problem_list = get_data(dataset=problem_name, language='kotlin')
        problem_list = self.data_unwrapper(raw_problem_list)
        method_list = list()
        for sample in tqdm(self.pipeline(problem_list, do_sample=True, top_k=top_k, temperature=0.1, top_p=0.95,
                                         num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id,
                                         max_length=500)):
            method_list.append(self.codellama_method_filter(sample[0]['generated_text']))
        file_name = f"output{problem_name}.json"
        method_dict = dict()
        for i, key in enumerate(raw_problem_list.keys()):
            method_dict[key] = method_list[i]
        with open(file_name, 'w') as f:
            json.dump(method_dict, f)
        self.method_dict = method_dict

    def codellama_process(self, problem_name: str):
        with jsonlines.open(f'output{problem_name}.jsonl', mode='w') as writer:
            for key, value in self.method_dict.items():
                answer_start = value.find('*/')
                body_start = value.find('\n', answer_start + 3)
                answer = value[body_start:]
                generated_sample = {"task_id": key, "completion": answer, "language": "kotlin"}
                writer.write(generated_sample)

    def codellama_evaluate(self, problem_name: str, model_outputs_jsonl: str, top_k=1, n_workers=8, timeout=15.0):
        if problem_name == "humaneval":
            reference_file = "./data/multilingual_humaneval/HumanEval_kotlin_v1.1.jsonl"
        elif problem_name == "mbkp":
            reference_file = "./data/mbxp/mbkp_release_v1.2.jsonl"
        else:
            raise NotImplementedError
        mxeval.evaluation.evaluate_functional_correctness(
            sample_file=model_outputs_jsonl, k=[top_k], n_workers=n_workers, timeout=timeout,
            problem_file=reference_file)
        with jsonlines.open('output.jsonl_results.jsonl') as reader:
            total_samples = 0
            passed_samples = 0
            test_failed_samples = 0
            compilation_error_samples = 0
            for obj in reader:
                total_samples += 1
                if obj["passed"]:
                    passed_samples += 1
                elif "Exception" in obj["result"]:
                    assert obj["time_elapsed"] > 0.0
                    test_failed_samples += 1
                elif "error" in obj["result"]:
                    assert ".kt" in obj["result"]
                    assert obj["time_elapsed"] is None
                    compilation_error_samples += 1
                else:
                    raise Exception("Unexpected behavior from mxeval")
            pass_rate = round(100.0 * passed_samples / total_samples, 2)
            test_fail_rate = round(100.0 * test_failed_samples / total_samples, 2)
            compilation_error_rate = round(100.0 * compilation_error_samples / total_samples, 2)
        self.method_dict["pass_rate"] = pass_rate
        self.method_dict["test_fail_rate"] = test_fail_rate
        self.method_dict["compilation_error_rate"] = compilation_error_rate
        with open(f"output{problem_name}.json", 'w') as f:
            json.dump(self.method_dict, f)
        return f"pass rate: {pass_rate}, test fail rate: {test_fail_rate}, compilation error rate: {compilation_error_rate}"





