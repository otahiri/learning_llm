"""
neural_network program
educational script to make nn from 0
"""

import numpy as np


def main():
    input_data = np.array([[ 1, 2, 3, 2.5 ],
                           [1, 2, 3, 2.5]])
    weight = np.array([[
            [0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, 0.27, 0.17, 0.87]
          ],
                       [
            [0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, 0.27, 0.17, 0.87]
          ]])
    biases = np.array([2, 3, 0.5])
    layer_output = np.dot(weight, input_data) + biases
    print(layer_output.shape)


if __name__ == "__main__":
    main()
