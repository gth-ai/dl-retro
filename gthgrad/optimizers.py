import math
from engine import Value


class SGD:
    """
    Stochastic Gradient Descent optimizer.
    """
    def __init__(self, parameters, lr=0.01,momentum=0.0):
        """
        Initializes the SGD optimizer.

        Args:
            parameters (list of Value): Parameters to optimize.
            lr (float, optional): Learning rate.
            momentum: maintain direction of the optimization
        """
        self.parameters = parameters
        self.momentum = momentum
        self.lr = lr
        self.velocities = [0.0] + len(parameters)

    def step(self):
        """
        Performs a single optimization step.
        """
        for i,p in enumerate(self.parameters):
            self.velocities[i] = self.momentum * self.velocities[i] - self.lr * p.grad
            p.data += self.velocities[i]

    def zero_grad(self):
        """
        Resets the gradients of all parameters to zero.
        """
        for p in self.parameters:
            p.grad = 0.0

class Adam:
    def __init__(self):
        pass