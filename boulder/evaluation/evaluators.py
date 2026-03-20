from dataclasses import dataclass
from typing import Literal

import numpy as np

from boulder.evaluation.metrics import accuracy, mean_absolute_error, precision


@dataclass
class TaskResult:
    average_score: float
    scores: np.ndarray
    metric: Literal['accuracy', 'mae', 'precision']
    response_lengths: list[int]


def evaluate_time_task(items: list[dict]) -> TaskResult:
    """Evaluate sunset/time task: exact match accuracy."""
    answers = []
    targets = []
    response_lengths = []

    for example in items:
        answers.append(example.get("parsed_answer"))
        targets.append(example.get("target"))
        response_lengths.append(len(example.get("response", "")))

    exact_matches = [1.0 if p == t else 0.0 for p, t in zip(answers, targets)]

    return TaskResult(
        average_score=np.mean(exact_matches),
        scores=np.array(exact_matches),
        metric='accuracy',
        response_lengths=response_lengths,
    )


def evaluate_price_task(items: list[dict]) -> TaskResult:
    """Evaluate price task: integer accuracy."""
    answers = []
    targets = []
    response_lengths = []

    for example in items:
        parsed_answer = example.get("parsed_answer")
        target = example.get("target")
        if parsed_answer is None:
            parsed_answer = 0.0
        if isinstance(parsed_answer, str) and "\u00a3" in parsed_answer:
            parsed_answer = parsed_answer.replace("\u00a3", "")
        answers.append(float(parsed_answer))
        targets.append(float(target))
        response_lengths.append(len(example.get("response", "")))

    answers_arr = np.array(answers)
    targets_arr = np.array(targets)
    scores = accuracy(answers_arr, targets_arr)

    return TaskResult(
        average_score=scores.mean(),
        scores=scores,
        metric='accuracy',
        response_lengths=response_lengths,
    )


def evaluate_numeric_task(items: list[dict]) -> TaskResult:
    """Evaluate distance/frequency task: MAE."""
    answers = []
    targets = []
    response_lengths = []

    for example in items:
        parsed_answer = example.get("parsed_answer")
        if parsed_answer is None:
            parsed_answer = float('inf')
        answers.append(float(parsed_answer))
        targets.append(float(example.get("target")))
        response_lengths.append(len(example.get("response", "")))

    answers_arr = np.array(answers)
    targets_arr = np.array(targets)
    mae_by_example = mean_absolute_error(answers_arr, targets_arr)
    finite_mask = np.isfinite(mae_by_example)
    mae = float(mae_by_example[finite_mask].mean()) if finite_mask.any() else float('inf')

    return TaskResult(
        average_score=mae,
        scores=mae_by_example,
        metric='mae',
        response_lengths=response_lengths,
    )


def evaluate_path_task(items: list[dict]) -> TaskResult:
    """Evaluate path task: exact ordered list match."""
    accuracies_list = []
    response_lengths = []

    for example in items:
        parsed_answer = example.get("parsed_answer")
        if isinstance(parsed_answer, dict):
            parsed_answer = parsed_answer.get("optimal_order")
        target = example.get("target")

        if parsed_answer is None:
            parsed_answer = []
        if target is None:
            target = []
        if not isinstance(parsed_answer, list):
            parsed_answer = [parsed_answer]
        if not isinstance(target, list):
            target = [target]

        if len(parsed_answer) == len(target) and all(p.lower() == t.lower() for p, t in zip(parsed_answer, target)):
            accuracies_list.append(1.0)
        else:
            accuracies_list.append(0.0)

        response_lengths.append(len(example.get("response", "")))

    accuracies_array = np.array(accuracies_list)

    return TaskResult(
        average_score=accuracies_array.mean(),
        scores=accuracies_array,
        metric='accuracy',
        response_lengths=response_lengths,
    )


def evaluate_restaurants_task(items: list[dict]) -> TaskResult:
    """Evaluate restaurants task: precision of parsed restaurant names."""
    parsed_restaurants = []
    target_restaurants = []
    response_lengths = []

    for example in items:
        parsed_answer = example.get("parsed_answer")
        target = example.get("target")

        if parsed_answer is None:
            parsed_answer = []
        if not isinstance(parsed_answer, list):
            parsed_answer = [parsed_answer]
        parsed_answer = [x.lower() if isinstance(x, str) else x for x in parsed_answer]

        if target is None:
            target = []
        if not isinstance(target, list):
            target = [target]
        target_names = [x.get("name").lower() for x in target if isinstance(x, dict) and "name" in x]

        parsed_restaurants.append(parsed_answer)
        target_restaurants.append(target_names)
        response_lengths.append(len(example.get("response", "")))

    scores = precision(parsed_restaurants, target_restaurants)

    return TaskResult(
        average_score=scores.mean(),
        scores=scores,
        metric='precision',
        response_lengths=response_lengths,
    )


def evaluate_direction_task(items: list[dict]) -> TaskResult:
    """Evaluate direction task: yes/no/unknown string comparison."""
    accuracies_list = []
    response_lengths = []

    for example in items:
        parsed_answer = example.get("parsed_answer")
        target = example.get("target")

        if parsed_answer is None:
            parsed_answer = ""
        if isinstance(parsed_answer, str):
            parsed_answer = parsed_answer.lower().strip()

        if isinstance(target, bool):
            target = "yes" if target else "no"
        elif isinstance(target, str):
            target = target.lower().strip()
        else:
            target = ""

        accuracies_list.append(1.0 if parsed_answer == target else 0.0)
        response_lengths.append(len(example.get("response", "") or ""))

    accuracies_array = np.array(accuracies_list)

    return TaskResult(
        average_score=accuracies_array.mean(),
        scores=accuracies_array,
        metric='accuracy',
        response_lengths=response_lengths,
    )


TASK_EVALUATORS = {
    "times": evaluate_time_task,
    "amount": evaluate_price_task,
    "distance": evaluate_numeric_task,
    "frequency": evaluate_numeric_task,
    "order": evaluate_path_task,
    "restaurants": evaluate_restaurants_task,
    "directions": evaluate_direction_task,
}
