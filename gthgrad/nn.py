import random
from engine import Value

class Neuron:
    """
    Represents a single artificial neuron.

    Each neuron has a set of weights and a bias, and it applies a weighted sum followed
    by a non-linear activation function (tanh) to its inputs.
    """

    def __init__(self, nin):
        """
        Initializes a Neuron with a given number of inputs.

        Parameters:
        - nin (int): Number of inputs to the neuron.
        """
        # Initialize weights and bias with random values.
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        """
        Computes the output of the neuron for a given input.

        Parameters:
        - x (list of Value): Input values to the neuron.

        Returns:
        - Value: The output of the neuron after applying the tanh activation function.
        """
        # Compute the weighted sum of inputs plus the bias.
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()  # Apply the tanh activation function.

        return out

    def parameters(self):
        """
        Returns all learnable parameters of the neuron (weights and bias).

        Returns:
        - list of Value: List of all parameters.
        """
        return self.w + [self.b]


class Layer:
    """
    Represents a single layer of neurons in a neural network.

    A layer consists of multiple neurons, each processing the same inputs in parallel.
    """

    def __init__(self, nin, nout):
        """
        Initializes a Layer with the specified number of inputs and outputs.

        Parameters:
        - nin (int): Number of inputs to the layer.
        - nout (int): Number of neurons in the layer (outputs of the layer).
        """
        # Create a list of neurons for the layer.
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        """
        Computes the output of the layer for a given input.

        Parameters:
        - x (list of Value): Input values to the layer.

        Returns:
        - list of Value: Outputs of the layer, or a single Value if the layer has one neuron.
        """
        # Compute the output of each neuron in the layer.
        outs = [n(x) for n in self.neurons]

        return outs[0] if len(outs) == 1 else outs  # Return a single value if only one neuron.

    def parameters(self):
        """
        Returns all learnable parameters of the layer (weights and biases of all neurons).

        Returns:
        - list of Value: List of all parameters in the layer.
        """
        params = []

        # Collect parameters from all neurons.
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)

        return params


class MLP:
    """
    Represents a Multi-Layer Perceptron (MLP).

    An MLP is composed of multiple layers, each processing inputs sequentially.
    """

    def __init__(self, nin, nouts):
        """
        Initializes an MLP with the specified architecture.

        Parameters:
        - nin (int): Number of inputs to the MLP.
        - nouts (list of int): List specifying the number of neurons in each layer.
        """
        # Create a sequence of layers based on the architecture.
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        """
        Computes the output of the MLP for a given input.

        Parameters:
        - x (list of Value): Input values to the MLP.

        Returns:
        - Value or list of Value: Output of the MLP.
        """
        # Sequentially pass input through each layer.
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """
        Returns all learnable parameters of the MLP (weights and biases of all layers).

        Returns:
        - list of Value: List of all parameters in the MLP.
        """
        # Collect parameters from all layers.
        return [p for layer in self.layers for p in layer.parameters()]