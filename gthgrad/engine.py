import math

class Value:
    """
    Represents a scalar value in a computational graph.

    This class is inspired by micrograd and allows automatic differentiation
    through a series of mathematical operations using a computational graph.
    """

    def __init__(self, data, _children=(), _op='', label=''):
        """
        Initializes a Value object.

        Parameters:
        - data (float): The scalar value.
        - _children (tuple): Parent nodes contributing to this value in the graph.
        - _op (str): The operation that generated this value (e.g., '+', '*').
        - label (str): Optional label for the value.
        """
        self.data = data
        self._op = _op
        self.label = label

        # Gradient of this value (used in backpropagation).
        self.grad = 0.0

        # Function to compute gradients for parent nodes.
        self._backward = lambda: None 

        # Set of parent nodes in the computational graph.
        self._prev = set(_children)

    def __repr__(self):
        """
        Returns a string representation of the Value object.
        """
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        """
        Overloads the '+' operator for scalar addition.

        If `other` is not a Value, it is converted to one.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op='+')

        def _backward():
            # Gradients of addition are distributed equally.
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out 
    
    def __radd__(self, other):
        """
        Ensures addition is commutative by implementing reverse addition.
        """
        return self + other
        
    def __mul__(self, other):
        """
        Overloads the '*' operator for scalar multiplication.

        If `other` is not a Value, it is converted to one.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op='*')

        def _backward():
            # Gradient of multiplication involves the other operand.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __rmul__(self, other):
        """
        Ensures multiplication is commutative by implementing reverse multiplication.
        """
        return self * other
        
    def __truediv__(self, other):
        """
        Overloads the '/' operator for scalar division.
        """
        return self * other**(-1)
        
    def __pow__(self, other):
        """
        Overloads the '**' operator for power operations.

        Parameters:
        - other (Value or float): Exponent. If a Value, must be non-negative.
        """
        if isinstance(other, Value):
            if self.data < 0:
                raise ValueError("Negative bases are not supported when exponent is a Value")
            out = Value(self.data ** other.data, _children=(self, other), _op=f'**{other}')

            def _backward():
                # Gradient for power operation when the exponent is a Value.
                self.grad += other.data * (self.data**(other.data - 1)) * out.grad
                other.grad += (math.log(self.data) * out.data) * out.grad

            out._backward = _backward

            return out

        elif isinstance(other, (int, float)):
            if self.data < 0 and not isinstance(other, int):
                raise ValueError("Negative bases with non-integer exponents are not supported")
            if self.data == 0 and other <= 0:
                raise ValueError("0 cannot be raised to a non-positive power")
            
            out = Value(self.data ** other, _children=(self,), _op=f"**{other}")

            def _backward():
                # Gradient for power operation when the exponent is a constant.
                self.grad += other * (self.data**(other - 1)) * out.grad

            out._backward = _backward

            return out
    
    def __neg__(self):
        """
        Overloads the unary '-' operator for negation.
        """
        return self * (-1)
    
    def __sub__(self, other):
        """
        Overloads the '-' operator for subtraction.
        """
        return self + (-other)

    def tanh(self):
        """
        Computes the hyperbolic tangent (tanh) activation function.
        """
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, _children=(self,), _op='tanh')

        def _backward():
            # Gradient of tanh is 1 - tanh(x)^2.
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out
    
    def exp(self):
        """
        Computes the exponential function.
        """
        t = math.exp(self.data)
        out = Value(t, _children=(self,), _op='exp')
        
        def _backward():
            # Gradient of exp is the function itself.
            self.grad += t * out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        """
        Performs backpropagation to compute gradients for all nodes.

        Uses topological sorting to ensure proper order of computation.
        """
        topo = []  # Topologically sorted nodes
        visited = set()

        def build_topo(v):
            # Recursive function to build the topological order.
            if v not in visited:
                for child in v._prev:
                    build_topo(child)
                visited.add(v)
            topo.append(v)
        
        build_topo(self)  # Start building from the current node.
        self.grad = 1.0 # Initialize gradient

        for node in reversed(topo):
            node._backward()  # Apply the backward function.