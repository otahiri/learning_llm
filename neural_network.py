"""
neural_network program
educational script to make nn from 0
"""

import numpy as np


def segmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def derivative_seg(x):
    return x * (1 - x)


def main() -> None:
    """the entry point for the program"""
    neuron = 8
    passes = 10000
    learning_rate = 0.5

    input_data = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    expected_output = np.array([[0], [1], [1], [0]])
    w1 = np.random.randn(2, neuron) * 0.1
    b1 = np.random.randn(1, neuron) * 0.1
    w2 = np.random.randn(neuron, 1) * 0.1
    b2 = np.random.randn(1, 1) * 0.1
    for i in range(passes):
        hidden_linear = input_data @ w1 + b1
        hidden_activated = segmoid(hidden_linear)
        seg_linear = hidden_activated @ w2 + b2
        output = segmoid(seg_linear)

        error = output - expected_output
        d_output = error * derivative_seg(output)
        w2_gradient = hidden_activated.T @ d_output
        b2_gradient = np.sum(d_output, axis=0, keepdims=True)
        hidden_error = d_output @ w2.T
        d_hidden = hidden_error * derivative_seg(hidden_activated)
        w1_gradient = input_data.T @ d_hidden
        b1_gradient = np.sum(d_hidden, axis=0, keepdims=True)

        w1 = w1 - (learning_rate * w1_gradient)
        w2 = w2 - (learning_rate * w2_gradient)
        b1 = b1 - (learning_rate * b1_gradient)
        b2 = b2 - (learning_rate * b2_gradient)
    print(np.round(output, 0))

    input_data = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    hidden_linear = input_data @ w1 + b1
    hidden_activated = segmoid(hidden_linear)
    seg_linear = hidden_activated @ w2 + b2
    output = segmoid(seg_linear)
    
    print(output[0] + output[1])


if __name__ == "__main__":
    main()
