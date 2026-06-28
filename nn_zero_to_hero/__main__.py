from nn_zero_to_hero import Value


def main():
    a = Value(29.0, label='a')
    b = Value(2.0, label='b')
    d = a + b
    print(d)
    print(d)
    print(d._op)


if __name__ == "__main__":
    main()
