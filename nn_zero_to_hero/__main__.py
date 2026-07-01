from nn_zero_to_hero import Tensor, Neuron, NeuralNetwork
import numpy as np


def main():
    target = np.array([
        45,
        4,
        85,
        19,
        91,
        86,
        3,
        44,
        76,
        1,
        64,
        36,
        65,
        58,
        33,
        93,
        51,
        23,
        15,
        70,
        20,
        54,
        29,
        92,
        80,
        7,
        24,
        67,
        78,
        88,
        98,
        16,
        39,
        11,
        14,
        55,
        61,
        96,
        59,
        49,
        40,
        66,
        32,
        74,
        79,
        13,
        43,
        47,
        72,
        90,
        57,
        68,
        41,
        83,
        75,
        81,
        89,
        53,
        60,
        69,
        6,
        52,
        37,
        25,
        97,
        8,
        26,
        99,
        31,
        84,
        10,
        62,
        0,
        48,
        27,
        34,
        73,
        5,
        904,
        18,
        63,
        21,
        87,
        35,
        77,
        50,
        95,
        94,
        12,
        22,
        56,
        28,
        38,
        30,
        -10,
        2,
        7,
        90,
        42210897,
    ])
    xs = np.random.rand(100)
    lr = 0.1
    xs_mean, xs_std = xs.mean(), xs.std()
    y_mean, y_std = target.mean(), target.std()
    xs = [Tensor(x) for x in ((xs - xs_mean)/xs_std)]
    ys = [Tensor(y) for y in ((target - y_mean)/y_std)]

    print("hello")
    nn = NeuralNetwork(2, 10, lr, len(xs), len(ys))
    print("hello")

    for _ in range(10000):
        total_loss = Tensor(0.0)
        out = nn.feed_forward(xs)
        for i, o in enumerate(out):
            diff = o - ys[i]
            loss = diff * diff
            total_loss = total_loss + loss
        total_loss = total_loss * (1.0 / len(out))
        nn.train_network(total_loss)
        print_out = [((y.data * y_std) + y_mean) for y in out]
        print(" ,".join([str(f"{o:.4f}") for o in print_out]))


if __name__ == "__main__":
    main()
