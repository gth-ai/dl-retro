import math


class Value:
    """
    This class is my reimplementation of micrograd of AK.
    """


    # We want to initialize the Value object
    def __init__(self, data, _children=(),_op='', label=''):
        self.data = data
        self._op = _op
        self.label = label

        self.grad = 0.0
        self._backward = lambda:None
        self._prev = set(_children)

    def __repr__(self):
        return f"Value=(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self,other), '+')

        def _backward():
            other.grad += 1.0 + out.grad
            self.grad += 1.0 + out.grad
        out._backward = _backward

        return out 
    
    # the following fonction is to reinsure that addition op is commutative
    def __radd__(self,other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward(): # real implementation of _backward = lambda:None
            self.grad += out.grad * other.data
            other.grad += self.data * out.grad
        out._backward = _backward

        return out 
    
    # the following fonction is to reinsure that multiplication op is commutative
    def __rmul__(self,other):
        return self * other
    
    def __truediv__(self,other):
        return self * other**-1
    
    def __power__(self, other):
        assert isinstance(other, (int,float)), "Support int or float only"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other -1)) * out.grad
        out._backward = _backward

        return out
    

