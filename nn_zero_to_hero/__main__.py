from nn_zero_to_hero import Tensor, Neuron, NeuralNetwork
import numpy as np


def main():
    target = Tensor(99982189)
    xs = [Tensor(x) for x in np.random.rand(100)]
    lr = 1e-11

    print("hello")
    nn = NeuralNetwork(2, 10, lr, len(xs))
    print("hello")

    for _ in range(10000):
        out = nn.feed_forward(xs)
        diff = out - target
        loss = diff * diff
        nn.train_network(loss)
        print(out)


if __name__ == "__main__":
    main()
