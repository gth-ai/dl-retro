import math


class Value:
    """
    This class is my reimplementation of micrograd of AK.
    """


    # We want to initialize the Value object
    def __init__(self, data, _children=(), _op='',label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label

        self.grad = 0.0
        self._backward = lambda: None # We know that we need a function, but depend on the operation we have to define it.

    def __rep__(self):
        return f"Value(data={self.data})"
    
    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self,other), '+')

        def _backward():
            self.grad += 1.0 + out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

