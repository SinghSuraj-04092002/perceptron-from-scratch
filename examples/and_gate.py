"""Train a perceptron on the AND logic gate (linearly separable)."""

import numpy as np

from perceptron import Perceptron

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # AND truth table

if __name__ == "__main__":
    model = Perceptron(n_inputs=2, learning_rate=0.1)
    errors = model.train(X, y, epochs=10, verbose=True)

    print("\nFinal weights:", model.weights)
    print("Final bias:", model.bias)
    print("\nPredictions:")
    for inputs in X:
        print(inputs, "->", model.predict(inputs))

    print("\nConverged:" , "yes" if errors[-1] == 0 else "no")
