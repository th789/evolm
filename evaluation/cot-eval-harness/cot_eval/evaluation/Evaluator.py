import os, json, re
import random
from typing import List, Dict, Tuple
from collections import defaultdict
import ast

from cot_eval.utils.utils import timeout
from cot_eval.evaluation.utils.utils import last_boxed_only_string, remove_boxed
from cot_eval.evaluation.utils.math_verify_toolkit.latex import latex_equiv


class Evaluator:
    def __init__(self, answer_extraction_format, num_last_chars_for_eval=128) -> None:
        assert answer_extraction_format in [
            "boxed",
            "answer is",
            "both",
        ], f"Invalid answer extraction format {answer_extraction_format}."
        self.answer_extraction_format = answer_extraction_format
        self.num_last_chars_for_eval = num_last_chars_for_eval

    def validate_model_completion(self, completion: str) -> bool:
        if not isinstance(completion, str):
            return False

        if self.answer_extraction_format == "boxed":
            if "\\boxed" in completion:
                return True
            else:
                return False
        elif self.answer_extraction_format == "answer is":
            if "answer is" in completion:
                return True
            else:
                return False
        elif self.answer_extraction_format == "both":
            if "\\boxed" in completion or "answer is" in completion:
                return True
            else:
                return False
        else:
            raise ValueError(f"Invalid answer extraction format {self.answer_extraction_format}")

    def validate_model_answer(self, answer: str) -> bool:
        if answer is None:
            return False

        if not isinstance(answer, str):
            return False

        if len(answer) == 0:
            return False

        return True

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        """Judge whether two answers are equivalent."""
        if not self.validate_model_answer(answer_a) or not self.validate_model_answer(answer_b):
            return None

        @timeout(timeout_seconds=4)
        def check(x, y):
            return self._check_answers_equiv(x, y)

        def check_wrapper(x, y):
            try:
                return check(x, y)
            except TimeoutError:
                return None

        return any([check_wrapper(answer_a, answer_b), check_wrapper(answer_b, answer_a)])

    def _check_answers_equiv(self, answer_a: str, answer_b: str):
        raise NotImplementedError

    def extract_answer_from_gold_solution(self, solution: str) -> str:
        """Extract the answer from the gold solution."""
        return self._extract_answer_from_gold_solution(solution)

    def _extract_answer_from_gold_solution(self, solution: str) -> str:
        raise NotImplementedError

    def extract_answer_from_model_completion(self, completion: str) -> str:
        """Extract the answer from the model completion."""
        if not self.validate_model_completion(completion):
            return None

        return self._extract_answer_from_model_completion(completion)

    def _extract_answer_from_model_completion(self, completion: str) -> str:
        def extract_from_boxed(c):
            box = last_boxed_only_string(c)
            ans = remove_boxed(box)
            return ans if ans else None

        def extract_from_answer_is(c):
            split = c.split("answer is")
            if len(split) > 1:
                answer = split[-1].strip()
                if ":" in answer:
                    answer = answer.split(":")[-1].strip()
                return answer
            else:
                return None

        if self.answer_extraction_format == "boxed":
            return extract_from_boxed(completion)
        elif self.answer_extraction_format == "answer is":
            return extract_from_answer_is(completion)
        elif self.answer_extraction_format == "both":
            boxed_ans = extract_from_boxed(completion)
            if boxed_ans:
                return boxed_ans

            answer_is_ans = extract_from_answer_is(completion)
            if answer_is_ans:
                return answer_is_ans

            return None
        else:
            raise ValueError(f"Invalid answer extraction format {self.answer_extraction_format}")

    def find_majority_completion_and_answer(self, candidates):
        """
        Finds the majority answer among the candidates using the evaluator's equivalence check.

        Args:
            candidates (List[str]): A list of candidate answer strings.
        """
        if not candidates:
            return None, None, None

        if len(candidates) == 1:
            majority_completion = candidates[0]
            majority_answer = self.extract_answer_from_model_completion(
                majority_completion[-self.num_last_chars_for_eval :]
            )
            return majority_completion, majority_answer, 0

        clusters = []  # Each cluster is a list of equivalent answers

        for candidate in candidates:
            candidate_answer = self.extract_answer_from_model_completion(candidate[-self.num_last_chars_for_eval :])
            placed = False
            for cluster in clusters:
                representative_answer = self.extract_answer_from_model_completion(
                    cluster[0][-self.num_last_chars_for_eval :]
                )
                if self.check_answers_equiv(candidate_answer, representative_answer):
                    cluster.append(candidate)
                    placed = True
                    break
            if not placed:
                clusters.append([candidate])

        # Find the cluster with the maximum number of candidates
        majority_cluster = max(clusters, key=lambda cluster: len(cluster))
        majority_completion = majority_cluster[0]
        majority_answer = self.extract_answer_from_model_completion(
            majority_completion[-self.num_last_chars_for_eval :]
        )

        # Also return the index of the majority completion in the candidates list
        majority_index = candidates.index(majority_completion)

        return majority_completion, majority_answer, majority_index
    
    def remove_text_brackets(self, s):
        s = s.strip()
        head = "\\text{"
        tail = "}"
        if s.startswith(head) and s.endswith(tail):
            return s[len(head) : -len(tail)]
        else:
            return s


class MATHEvaluator(Evaluator):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def _check_answers_equiv(self, answer_a: str, answer_b: str):
        try:
            res = latex_equiv(answer_a, answer_b)
        except Exception as e:
            res = False

        return res

    def _extract_answer_from_gold_solution(self, solution: str):
        box = last_boxed_only_string(solution)
        ans = remove_boxed(box)
        return ans if ans else None


class GSM8KEvaluator(MATHEvaluator):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def _extract_answer_from_gold_solution(self, solution: str | float):
        if isinstance(solution, float):
            return str(solution)
        elif isinstance(solution, str):
            return solution.split("#### ")[-1].strip()
        else:
            raise ValueError("Invalid type of gold solution")


class SciBenchEvaluator(MATHEvaluator):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def _extract_answer_from_gold_solution(self, solution: str):
        return solution


class FOLIOEvaluator(Evaluator):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def parse_NLI_answer(self, answer: str):
        answer = answer.lower().strip()

        true_words = ["yes", "true", "correct"]
        false_words = ["no", "false", "incorrect"]
        uncertain_words = ["unknown", "unanswerable", "uncertain"]

        try:
            if any(x in answer for x in true_words):
                assert not any(x in answer for x in false_words), f"Answer contains both true and false words: {answer}"
                assert not any(
                    x in answer for x in uncertain_words
                ), f"Answer contains both true and uncertain words: {answer}"
                return "true"
            elif any(x in answer for x in false_words):
                assert not any(x in answer for x in true_words), f"Answer contains both false and true words: {answer}"
                assert not any(
                    x in answer for x in uncertain_words
                ), f"Answer contains both false and uncertain words: {answer}"
                return "false"
            elif any(x in answer for x in uncertain_words):
                assert not any(
                    x in answer for x in true_words
                ), f"Answer contains both uncertain and true words: {answer}"
                assert not any(
                    x in answer for x in false_words
                ), f"Answer contains both uncertain and false words: {answer}"
                return "uncertain"
            else:
                return None
        except AssertionError as e:
            return None

    def _check_answers_equiv(self, answer_a: str, answer_b: str):
        answer_a = self.parse_NLI_answer(answer_a)
        answer_b = self.parse_NLI_answer(answer_b)
        return answer_a is not None and answer_b is not None and answer_a == answer_b

    def _extract_answer_from_gold_solution(self, solution: str):
        return self.parse_NLI_answer(solution)


class MMLUEvaluator(Evaluator):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def _check_answers_equiv(self, answer_a: str, answer_b: str):
        return answer_a.lower() == answer_b.lower()

    def _extract_answer_from_gold_solution(self, solution: str):
        return solution


class CRUXEvaluator(Evaluator):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def load_obj_as_str(self, s):
        try:
            return ast.literal_eval(s)
        except:
            return s

    def _check_answers_equiv(self, answer_a: str, answer_b: str):
        def check(x, y):
            return self.load_obj_as_str(x) == self.load_obj_as_str(y)

        def normalize(x):
            x = self.remove_text_brackets(x)
            if x.startswith("'") and x.endswith("'"):
                x = x[1:-1]
            return x
    
        return check(normalize(answer_a), normalize(answer_b))

    def _extract_answer_from_gold_solution(self, solution: str):
        return solution


class ZebraLogicEvaluator(Evaluator):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def _check_answers_equiv(self, answer_a: str, answer_b: str):
        return self.remove_text_brackets(answer_a) == self.remove_text_brackets(answer_b)

    def _extract_answer_from_gold_solution(self, solution: str):
        return solution


def get_evaluator(task_name: str, answer_extraction_format: str) -> Evaluator:
    if any(
        x in task_name
        for x in ["MATH", "OlympiadBench", "AMC", "AIME", "Minerva", "augmented_GSM8K_MATH_SFT-unfiltered"]
    ):
        return MATHEvaluator(answer_extraction_format)
    elif any(x in task_name for x in ["GSM"]):
        return GSM8KEvaluator(answer_extraction_format)
    elif any(x in task_name for x in ["FOLIO", "BoardgameQA", "StrategyQA"]):
        return FOLIOEvaluator(answer_extraction_format)
    elif any(x in task_name for x in ["MMLU", "GPQA"]):
        return MMLUEvaluator(answer_extraction_format)
    elif any(x in task_name for x in ["SciBench", "TableBench", "TabMWP", "HLE"]):
        return SciBenchEvaluator(answer_extraction_format)
    elif any(x in task_name for x in ["CRUX"]):
        return CRUXEvaluator(answer_extraction_format)
    elif any(x in task_name for x in ["ZebraLogic", "CommonsenseQA"]):
        return ZebraLogicEvaluator(answer_extraction_format)
    else:
        raise ValueError(f"Task name {task_name} not found in the evaluator mapping")
