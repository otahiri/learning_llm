from nn_zero_to_hero import Tensor
import numpy as np


class Neuron():
    def __init__(self, lr: float) -> None:
        self.lr = lr
        self.bias: Tensor
        self.weights: list = []

    def initiate_params(self, nin: int) -> None:
        rand_val = np.random.rand(nin)
        self.weights = [Tensor(x) for x in rand_val]
        self.bias = Tensor(np.random.random())

    def feed_forward(self, inputs: list) -> Tensor:
        total = Tensor(0)
        for w, x in zip(self.weights, inputs):
            total += w * x
        output = total + self.bias
        return output.ReLu()

    def back_propogation(self, target: Tensor, total: Tensor):
        diff = total - target
        loss = diff * diff
        loss.backwards()
        for w in self.weights:
            w.data -= self.lr * w.grad
            w.grad = 0.0
        self.bias.data -= self.lr * self.bias.grad
        self.bias.grad = np.array(0.0)
