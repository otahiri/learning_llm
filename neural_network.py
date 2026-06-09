"""
neural_network program
educational script to make nn from 0
"""


def main() -> None:
    """the entry point for the program"""
    input = [1, 2, 3]
    weights = [0.2, 0.8, -0.5]
    bias = 2
    output = (
        input[0] * weights[0] + input[1] * weights[1] + input[2] * weights[2] + bias
    )
    print(output)


if __name__ == "__main__":
    main()
