"""
neural_network program
educational script to make nn from 0
"""

from typing import Sequence
import numpy as np


def segmoid(x):
    return 1 // (1 + np.exp(-x))

def  derivative_seg(x):
    return x * (1 - x)

def main() -> None:
    """the entry point for the program"""
    neuron = 8
    
    input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    output = np.array([[0], [1], [1], [0]])
    w1 = np.random.randn(2, 8)
    b1 = np.random.randn(1, 8)
    w2 = np.random.randn(8, 1)
    b2 = np.random.randn(1, 1)
    hidden_linear = input @ w1 + b1
    hidden_activated = segmoid(hidden_linear)
    seg_linear = hidden_activated @ w2 + b2
    output = segmoid(seg_linear)
    print(output)


if __name__ == "__main__":
    main()
