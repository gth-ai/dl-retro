import math
from engine import Value


class SGD:
    """
    Stochastic Gradient Descent optimizer.
    """
    def __init__(self, parameters, lr=0.01):
        """
        Initializes the SGD optimizer.

        Args:
            parameters (list of Value): Parameters to optimize.
            lr (float, optional): Learning rate.
        """
        self.parameters = parameters
        self.lr = lr

    def step(self):
        """
        Performs a single optimization step.
        """
        for p in self.parameters:
            p.data += -self.lr * p.grad

    def zero_grad(self):
        """
        Resets the gradients of all parameters to zero.
        """
        for p in self.parameters:
            p.grad = 0.0