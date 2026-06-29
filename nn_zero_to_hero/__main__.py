from nn_zero_to_hero import Value, Neuron
import numpy as np


def main():
    target = Value(382)
    xs = [Value(x) for x in np.random.rand(100)]
    lr = 5e-4
    neuron = Neuron(lr)
    neuron.initiate_params(xs)

    for step in range(401):
        o = neuron.feed_forward(xs)
        neuron.back_propogation(target, o)
        if step % 10 == 0:
            print(o)


if __name__ == "__main__":
    main()
