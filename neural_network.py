from re import error
import numpy as np


def segmoid(x: float):
    return 1 / (1 + np.exp(-x))


def segmoid_derivative(a: float):
    return a * (1 - a)


def main():
    x = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
    y = np.array([
        [0],
        [1],
        [1],
        [0]
        ])
    input_size = 2
    hidden_size = 3
    output_size = 1
    learning_rate = 0.5
    np.random.seed(42)

    W1 = np.random.uniform(size=(input_size, hidden_size))
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.uniform(size=(hidden_size, output_size))
    b2 = np.zeros((1, output_size))
    for i in range(10000):
        Z1 = np.dot(x, W1) + b1
        A1 = segmoid(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = segmoid(Z2)
        error_output = A2 - y
        loss = np.mean(0.5 * (A2 - y) ** 2)
        dZ2 = error_output * segmoid_derivative(A2)
        error_hidden = dZ2.dot(W2.T)
        dZ1 = error_hidden * segmoid_derivative(A1)

        dW2 = A1.T.dot(dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dW1 = x.T.dot(dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        if i % 2000 == 0:
            print(f"Epoch {i:5d} | Loss: {loss:.6f}")
    print(np.round(A2))


if __name__ == "__main__":
    main()
