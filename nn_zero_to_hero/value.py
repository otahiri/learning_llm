import numpy as np


class Tensor:
    def __init__(self, data, _children=(), _op="", label="") -> None:
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._prev = tuple(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data * other.data, (self, other), "*")

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (other * Tensor(-1.0))

    def ReLu(self):
        out = Tensor(np.maximum(0.001 * self.data, self.data), (self,), "ReLu")
        return out

    def backwards(self):
        """
        build topoligical sort for the graph and calculate
        the gradiant for every node in the graph
        """
        self.grad = np.array(1.0)
        visited = set()
        topo = []

        def build_links(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_links(child)
                topo.append(v)

        build_links(self)
        for node in reversed(topo):
            children = list(node._prev)

            match node._op:
                case "ReLu":
                    children[0].grad += (
                        1.0 if children[0].data >= 0 else 0.001
                    ) * node.grad
                case "+":
                    for child in children:
                        child.grad += node.grad

                case "*":
                    children[1].grad += children[0].data * node.grad
                    children[0].grad += children[1].data * node.grad
