from sys import _current_exceptions

from nn_zero_to_hero import Tensor
import numpy as np


class Neuron:
    def __init__(self, lr: float) -> None:
        self.lr = lr
        self.weights: list = []

    def initiate_params(self, nin: int) -> None:
        rand_val = np.random.randn(nin)
        self.weights = [Tensor(x) for x in rand_val]
        self.bias = Tensor(np.random.randn())

    def feed_forward(self, inputs: list, hidden_layer: bool) -> Tensor:
        total = Tensor(0)
        for w, x in zip(self.weights, inputs):
            total += w * x
        output = total + self.bias
        return output.ReLu() if hidden_layer else output

    def parameters(self):
        return self.weights + [self.bias]


class NeuralNetwork:
    def __init__(self, hidden_layers, neurons, lr, nin: int, nout: int) -> None:
        self.neurons = neurons
        self.lr = lr
        self.layers: list[list[Neuron]] = []
        current_nin = nin
        for _ in range(hidden_layers):
            layer = [Neuron(lr) for _ in range(neurons)]
            for n in layer:
                n.initiate_params(current_nin)
            current_nin = neurons
            self.layers.append(layer)
        self.output = [Neuron(lr) for _ in range(nout)]
        for n in self.output:
            n.initiate_params(current_nin)

    def feed_forward(self, inputs: list):
        out = inputs
        for layer in self.layers:
            out = [n.feed_forward(out, True) for n in layer]
        return [n.feed_forward(out, False) for n in self.output]

    def parameters(self) -> list[Tensor]:
        params = []
        for layer in self.layers:
            for n in layer:
                params.extend(n.parameters())
        for n in self.output:
            params.extend(n.parameters())
        return params

    def train_network(self, loss: Tensor):
        loss.backwards()
        for p in self.parameters():
            p.data -= self.lr * p.grad
            p.grad = np.zeros_like(p.data, dtype=np.float64)
