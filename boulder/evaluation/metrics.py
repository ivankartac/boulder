from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    @abstractmethod
    def compute(self, predicted, target) -> np.ndarray:
        pass

    def __call__(self, predicted, target, reduce_mean: bool = False):
        values = self.compute(predicted, target)
        if reduce_mean:
            return values.mean()
        return values


class MeanSquaredError(Metric):
    def compute(self, predicted, target) -> np.ndarray:
        return (predicted - target) ** 2


class MeanAbsoluteError(Metric):
    def compute(self, predicted, target) -> np.ndarray:
        return np.abs(predicted - target)


class Accuracy(Metric):
    def __init__(self, integer_only: bool = False, tolerance: float = 0.01):
        self.integer_only = integer_only
        self.tolerance = tolerance

    def compute(self, predicted, target) -> np.ndarray:
        if self.integer_only:
            predicted = np.round(predicted).astype(int)
            target = np.round(target).astype(int)
        if self.tolerance > 0.01:
            predicted = np.round(predicted, 1)
        elif self.tolerance > 0.001:
            predicted = np.round(predicted, 2)
        return np.isclose(predicted, target, atol=self.tolerance).astype(float)


class Precision(Metric):
    def compute(self, predicted: list[list[str]], target: list[list[str]]) -> np.ndarray:
        result = []
        for p, t in zip(predicted, target):
            if len(p) == 0 and len(t) == 0:
                result.append(1.0)
            elif len(p) == 0:
                result.append(0.0)
            else:
                result.append(sum([x in t for x in p]) / len(p))
        return np.array(result)


mean_squared_error = MeanSquaredError()
mean_absolute_error = MeanAbsoluteError()
accuracy = Accuracy(integer_only=True)
precision = Precision()
