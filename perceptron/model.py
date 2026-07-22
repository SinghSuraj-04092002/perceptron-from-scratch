"""A single-layer perceptron implemented from scratch with NumPy."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class Perceptron:
    """Binary classifier using the classic perceptron learning rule.

    Attributes:
        weights: Learned weight vector, one entry per input feature.
        bias: Learned bias (threshold offset).
        lr: Learning rate used during training.
    """

    def __init__(self, n_inputs: int, learning_rate: float = 0.1) -> None:
        self.weights: NDArray[np.float64] = np.zeros(n_inputs)
        self.bias: float = 0.0
        self.lr: float = learning_rate

    @staticmethod
    def activation(x: float) -> int:
        """Step activation function: 1 if x >= 0, else 0."""
        return 1 if x >= 0 else 0

    def predict(self, inputs: NDArray[np.float64]) -> int:
        """Compute the perceptron's output for a single input vector."""
        weighted_sum = float(np.dot(inputs, self.weights) + self.bias)
        return self.activation(weighted_sum)

    def predict_batch(self, X: NDArray[np.float64]) -> NDArray[np.int_]:
        """Compute outputs for a batch of input vectors."""
        return np.array([self.predict(row) for row in X])

    def train(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int_],
        epochs: int = 10,
        verbose: bool = False,
    ) -> list[int]:
        """Train using the perceptron learning rule.

        Returns:
            A list of the number of misclassifications per epoch, useful
            for checking convergence (or lack thereof, e.g. on XOR).
        """
        errors_per_epoch: list[int] = []

        for epoch in range(epochs):
            total_errors = 0
            for inputs, target in zip(X, y):
                prediction = self.predict(inputs)
                error = target - prediction
                self.weights += self.lr * error * inputs
                self.bias += self.lr * error
                total_errors += int(error != 0)

            errors_per_epoch.append(total_errors)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - misclassifications: {total_errors}")

        return errors_per_epoch
