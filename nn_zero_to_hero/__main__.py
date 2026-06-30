from nn_zero_to_hero import Tensor, Neuron, NeuralNetwork
import numpy as np


def main():
    target = [
        Tensor(1),
        Tensor(87),
        Tensor(54),
        Tensor(5),
        Tensor(11),
        Tensor(3),
        Tensor(74),
        Tensor(53),
        Tensor(67),
        Tensor(94),
        Tensor(99),
        Tensor(60),
        Tensor(30),
        Tensor(81),
        Tensor(32),
        Tensor(79),
        Tensor(49),
        Tensor(90),
        Tensor(43),
        Tensor(66),
        Tensor(84),
        Tensor(52),
        Tensor(89),
        Tensor(64),
        Tensor(96),
        Tensor(37),
        Tensor(80),
        Tensor(50),
        Tensor(55),
        Tensor(77),
        Tensor(12),
        Tensor(40),
        Tensor(91),
        Tensor(33),
        Tensor(25),
        Tensor(86),
        Tensor(59),
        Tensor(10),
        Tensor(7),
        Tensor(0),
        Tensor(35),
        Tensor(48),
        Tensor(31),
        Tensor(8),
        Tensor(21),
        Tensor(88),
        Tensor(41),
        Tensor(92),
        Tensor(61),
        Tensor(45),
        Tensor(17),
        Tensor(38),
        Tensor(65),
        Tensor(22),
        Tensor(16),
        Tensor(68),
        Tensor(76),
        Tensor(57),
        Tensor(6),
        Tensor(69),
        Tensor(56),
        Tensor(26),
        Tensor(20),
        Tensor(13),
        Tensor(62),
        Tensor(36),
        Tensor(93),
        Tensor(24),
        Tensor(78),
        Tensor(46),
        Tensor(83),
        Tensor(95),
        Tensor(27),
        Tensor(23),
        Tensor(85),
        Tensor(71),
        Tensor(14),
        Tensor(98),
        Tensor(58),
        Tensor(51),
        Tensor(15),
        Tensor(42),
        Tensor(4),
        Tensor(70),
        Tensor(73),
        Tensor(9),
        Tensor(28),
        Tensor(47),
        Tensor(2),
        Tensor(97),
        Tensor(72),
        Tensor(63),
        Tensor(75),
        Tensor(39),
        Tensor(18),
        Tensor(29),
        Tensor(19),
        Tensor(34),
        Tensor(44),
        Tensor(82),
        Tensor(10),
        Tensor(10),
        Tensor(10),
    ]
    xs = [Tensor(x) for x in np.random.rand(100)]
    lr = 1e-5

    print("hello")
    nn = NeuralNetwork(2, 10, lr, len(xs), len(target))
    print("hello")

    for _ in range(10000):
        total_loss = Tensor(0.0)
        out = nn.feed_forward(xs)
        for i, o in enumerate(out):
            diff = o - target[i]
            loss = diff * diff
            total_loss = total_loss + loss
        nn.train_network(total_loss)
        print(" ,".join([str(f"{o.data:.4f}") for o in out]))


if __name__ == "__main__":
    main()
