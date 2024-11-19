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
        self._backward = lambda:None # depends on the operation, we have differents strategies of backward
        self._prev = set(_children)

    def __repr__(self):
        return f"Value=(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data, _children=(self,other), _op = '+')

        def _backward():
            self.grad += 1.0 * out.grad # grad of a + b = 1, because we considere b as constant
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out 
    
    # the following fonction is to reinsure that addition op is commutative
    def __radd__(self,other):
        return self + other
        
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self,other), _op='*')

        def _backward():
            self.grad += other.data * out.grad # grad from a * b considering b (other) as constant
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
        
    
    # the following fonction is to reinsure that multiplication op is commutative
    def __rmul__(self, other):
        return self * other
        
    
    def __truediv__(self,other):
        return self * other**(-1)
        
    
    def __pow__(self,other):
        if isinstance(other,Value):
            if self.data < 0:
                raise ValueError("Negative bases are not supported when exponent is a Value")
            out = Value(self.data ** other.data, _children=(self,other), op=f'**{other}')

            def _backward():
                self.grad += other.data*(self.data**(other.data-1)) * out.grad
                other.grad += (math.log(self.data)*out.data) * out.grad

            out._backward = _backward

            return out

        elif isinstance(other, (int,float)):
            if self.data < 0 and not isinstance(other,int):
                raise ValueError("Negative bases with non-integer exponents are not supported")
            if self.data.data == 0 and other <= 0:
                raise ValueError("0 cannot be raised to a non-positive power")
            
            out = Value(self.data**other, _children=(self,), _op=f"**{other}")

            def _backward():
                self.grad += other*(self.data**(other-1)) * out.grad

            out._backward = _backward

            return out
    
    def __neg__(self):
        return self * (-1)
    
    def __sub__(self,other): # soustraction
        return self + (-other) 

    def tanh(self):
        x = self.data
        t = ((math.exp(x)**2) - 1)/((math.exp(x)**2) + 1) # or (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, _children=(self,), _op='tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out
    
    def exp(self):
        t = math.exp(self.data)
        out = Value(t, _children=(self,), _op='exp')
        
        def _backward():
            self.grad += t * out.grad # or out.grad * out.data 
        out._backward = _backward

        return out
    
    def backward(self): #general backward for the wall network
        # we use the topological sort to implement the wall backward of our network

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                for child in v._prev:
                    build_topo(child)
                visited.add(v)
            topo.append(v)
        
        build_topo(self)
        for node in reversed(topo):
            node._backward()



    

    

