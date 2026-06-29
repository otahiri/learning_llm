from nn_zero_to_hero import Value, value


def main():
    x1 = Value(2.0, label="x1")
    x2 = Value(14.2, label="x2")

    w1 = Value(2.0, label="w1")
    w2 = Value(1.0, label="w2")
    b = Value(3.0, label="b")

    target = Value(5.92, label="target")
    x1w1 = Value(0, label="x1w1")
    x2w2 = Value(0, label="x2w2")
    x1w1x2w2 = Value(0, label="x1w1x2w2")
    n = Value(0, label="n")
    o = Value(1, label="o")

    for step in range(10000):
        x1w1 = x1 * w1
        x2w2 = x2 * w2
        x1w1x2w2 = x1w1 + x2w2
        n = x1w1x2w2 + b
        o = n.ReLu()
        diff = o - target
        loss = diff * diff
        loss.backwards()
        w1.data -= 0.1 * w1.grad
        w2.data -= 0.1 * w2.grad
        b.data -= 0.1 * b.grad

        w1.grad = 0.0
        w2.grad = 0.0
        b.grad = 0.0
        if step%100 == 0:
            print(o)



if __name__ == "__main__":
    main()
