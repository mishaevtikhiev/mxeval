from transformers import AutoTokenizer
import transformers
import torch
import mxeval
import json
import jsonlines
from tqdm import tqdm

from mxeval.data import get_data


class KotlinLLMEval:
    def __init__(self, model_name = "codellama/CodeLlama-7b-hf", gpu_id=0, method_dict_location:str=None):
        if method_dict_location is not None:
            with open(method_dict_location) as f:
                self.method_dict = json.load(f)
        else:
            self.method_dict = None
        self.model_name = ''
        if model_name is not None:
           self._init_model(model_name, gpu_id)

    def _init_model(self, model_name : str, gpu_id : int):
        pass

    def _data_unwrapper(self, problem_list: dict[str, dict]):
        for key in problem_list.keys():
            yield problem_list[key]['prompt']

    def model_generate(self, problem_list, top_k = 1, max_length = 500):
        pass

    def unwrap_model_output(self, problem_list: dict[str, dict]):
        pass
    def method_filter(self, answer: str):
        pass

    def generate(self, problem_name: str = 'multi-humaneval', top_k = 1, max_length = 500):
        if problem_name != 'multi-humaneval' and problem_name != 'mbxp':
            raise Exception('This dataset is not implemented')
        raw_problem_list = get_data(dataset=problem_name, language='kotlin')
        problem_list = self._data_unwrapper(raw_problem_list)
        method_list = self.model_generate(problem_list, top_k, max_length)
        file_name = f"output_{problem_name}_{self.model_name}.json"
        method_dict = dict()
        for i, key in enumerate(raw_problem_list.keys()):
            method_dict[key] = method_list[i]
        with open(file_name, 'w') as f:
            json.dump(method_dict, f)
        self.method_dict = method_dict

    def model_process(self, problem_name: str = 'multi-humaneval'):
        pass

    def evaluate(self, problem_name: str, model_outputs_jsonl: str, top_k=1, n_workers=8, timeout=15.0):
        if problem_name == "humaneval":
            reference_file = "./data/multilingual_humaneval/HumanEval_kotlin_v1.1.jsonl"
        elif problem_name == "mbkp":
            reference_file = "./data/mbxp/mbkp_release_v1.2.jsonl"
        else:
            raise NotImplementedError
        mxeval.evaluation.evaluate_functional_correctness(
            sample_file=model_outputs_jsonl, k=[top_k], n_workers=n_workers, timeout=timeout,
            problem_file=reference_file)
        model_execution_results = model_outputs_jsonl + '_results.jsonl'
        with jsonlines.open(model_execution_results) as reader:
            total_samples = 0
            passed_samples = 0
            test_failed_samples = 0
            compilation_error_samples = 0
            out_of_time_samples = 0
            for obj in reader:
                total_samples += 1
                if obj["passed"]:
                    passed_samples += 1
                elif "Exception" in obj["result"]:
                    assert obj["time_elapsed"] > 0.0
                    test_failed_samples += 1
                elif "error" in obj["result"]:
                    if not ".kt" in obj["result"]:
                        print(obj["result"])
                    if obj["time_elapsed"] is not None:
                        print(obj["result"])
                    compilation_error_samples += 1
                elif "timed out" in obj["result"]:
                    out_of_time_samples += 1
                else:
                    #raise Exception("Unexpected behavior from mxeval")
                    print("Unexpected behavior from mxeval")
                    print(obj["result"])

        pass_rate = round(100.0 * passed_samples / total_samples, 2)
        test_fail_rate = round(100.0 * test_failed_samples / total_samples, 2)
        compilation_error_rate = round(100.0 * compilation_error_samples / total_samples, 2)
        out_of_time_rate = round(100 * out_of_time_samples / total_samples, 2)
        self.method_dict["pass_rate"] = pass_rate
        self.method_dict["test_fail_rate"] = test_fail_rate
        self.method_dict["compilation_error_rate"] = compilation_error_rate
        self.method_dict["out_of_time_rate"] = out_of_time_rate
        with open(f"output_{problem_name}_{self.model_name}.json", 'w') as f:
            json.dump(self.method_dict, f)
        return f"pass rate: {pass_rate}, test fail rate: {test_fail_rate}, compilation error rate: {compilation_error_rate}, out of time rate: {out_of_time_rate}"





