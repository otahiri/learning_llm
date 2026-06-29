from nn_zero_to_hero import Tensor, Neuron
import numpy as np


def main():
    target = Tensor(3.2)
    xs = [Tensor(x) for x in np.random.rand(100)]
    lr = 5e-4
    neuron = Neuron(lr)
    neuron.initiate_params(len(xs))

    for step in range(400):
        o = neuron.feed_forward(xs)
        neuron.back_propogation(target, o)
        if step % 10 == 0:
            print(o)


if __name__ == "__main__":
    main()
