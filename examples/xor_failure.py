"""Demonstrate that a single perceptron cannot learn XOR.

XOR is not linearly separable, so the perceptron learning rule will
never converge to zero misclassifications. This script trains on XOR
and plots the misclassification count per epoch to show it oscillating
rather than settling at zero.
"""

import numpy as np
import matplotlib.pyplot as plt

from perceptron import Perceptron

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # XOR truth table

if __name__ == "__main__":
    model = Perceptron(n_inputs=2, learning_rate=0.1)
    errors = model.train(X, y, epochs=50, verbose=True)

    print("\nFinal weights:", model.weights)
    print("Final bias:", model.bias)
    print("\nPredictions (never fully correct):")
    for inputs in X:
        print(inputs, "->", model.predict(inputs))

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(errors) + 1), errors, marker="o")
    plt.title("Perceptron on XOR: misclassifications never reach zero")
    plt.xlabel("Epoch")
    plt.ylabel("Misclassifications")
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("xor_failure.png", dpi=150)
    print("\nSaved plot to xor_failure.png")
