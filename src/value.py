import math

# The autograd engine. Each Value is a node in a computation graph
# Forward pass: compute the results, track how they were computed
# Backward pass: walk the graph in reverse, applying the chain rule to get gradients

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    # core ops: each returns a new Value tracking its inputs + local gradients

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # d(a+b)/da = 1, d(a+b)/db = 1
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # d(a*b)/da = b, d(a*b)/db = a
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    # d(a^n)/da = n * a^(n-1)
    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    # d(ln(a))/da = 1/a
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    # d(e^a)/da = e^a
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    # d(relu(a))/da = 1 if a > 0 else 0
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))

    # --- convenience ops: all defined in terms of the core ops above ---
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
