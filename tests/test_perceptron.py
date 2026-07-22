import numpy as np
import pytest

from perceptron import Perceptron


def test_predict_output_is_binary():
    model = Perceptron(n_inputs=2)
    result = model.predict(np.array([1.0, 1.0]))
    assert result in (0, 1)


def test_and_gate_converges():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    model = Perceptron(n_inputs=2, learning_rate=0.1)
    errors = model.train(X, y, epochs=10)

    assert errors[-1] == 0
    for inputs, expected in zip(X, y):
        assert model.predict(inputs) == expected


def test_xor_does_not_converge():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    model = Perceptron(n_inputs=2, learning_rate=0.1)
    errors = model.train(X, y, epochs=50)

    # A single perceptron can never fully solve XOR
    assert errors[-1] != 0


def test_predict_batch_shape():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    model = Perceptron(n_inputs=2)
    predictions = model.predict_batch(X)
    assert predictions.shape == (4,)
