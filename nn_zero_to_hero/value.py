class Value:
    def __init__(self, data, _children=(), _op="", label="") -> None:
        self.data = data
        self.grad = 0.0
        self._prev = _children
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), "*")

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (other * Value(-1.0))

    def ReLu(self):
        out = Value(max(0.001 * self.data, self.data), (self,), "ReLu")
        return out

    def __pow__(self, pow):
        return Value(self.data ** pow, (self,), "pow")

    def backwards(self):
        self.grad = 1.0
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
