from nn_zero_to_hero import Value
import numpy as np


class Neuron():
    def __init__(self, lr: float) -> None:
        self.lr = lr
        self.bias: Value
        self.weights: list = []

    def initiate_params(self, inputs: list) -> None:
        rand_val = np.random.rand(len(inputs))
        self.weights = [Value(x) for x in rand_val]
        self.bias = Value(np.random.random())

    def feed_forward(self, inputs: list) -> Value:
        total = Value(0)
        for w, x in zip(self.weights, inputs):
            total += w * x
        output = total + self.bias
        return output.ReLu()

    def back_propogation(self, target: Value, total: Value):
        diff = total - target
        loss = diff * diff
        loss.backwards()
        for w in self.weights:
            w.data -= self.lr * w.grad
            w.grad = 0
        self.bias.data -= self.lr * self.bias.grad
        self.bias.grad = 0
