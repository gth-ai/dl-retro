import random
from engine import Value

class Neuron:
    """
    Represents a single neuron in a neural network.
    """
    def __init__(self, nin, activation='tanh'):
        """
        Initializes a Neuron.

        Args:
            nin (int): Number of input connections.
            activation (str, optional): Activation function ('tanh', 'relu', 'sigmoid', 'leaky_relu').
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation

    def __call__(self, x):
        """
        Computes the neuron's output for a given input.

        Args:
            x (list of Value): Input values.

        Returns:
            Value or list of Value: Activated output.
        """
        act =  sum((wi*xi for wi,xi in zip(self.w,x)),self.b)

        if self.activation == 'tanh':
            out = act.tanh()
        elif self.activation == 'relu':
            out = act.relu()
        elif self.activation == 'sigmoid':
            out = act.sigmoid()
        elif self.activation == 'leaky_relu':
            out = act.leaky_relu()
        else:
            out = act  # No activation
        return out

    def parameters(self):
        """
        Returns the parameters of the neuron.

        Returns:
            list of Value: Weights and bias.
        """
        return self.w + [self.b]


class Layer:
    """
    Represents a single layer of neurons in a neural network.

    A layer consists of multiple neurons, each processing the same inputs in parallel.
    """

    def __init__(self, nin, nout,activation='tanh'):
        """
        Initializes a Layer with the specified number of inputs and outputs.

        Parameters:
        - nin (int): Number of inputs to the layer.
        - nout (int): Number of neurons in the layer (outputs of the layer).
        """
        # Create a list of neurons for the layer.
        self.neurons = [Neuron(nin,activation) for _ in range(nout)]

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
    Multi-Layer Perceptron model consisting of multiple layers.
    """
    def __init__(self, nin, nouts, activation='tanh'):
        """
        Initializes an MLP.

        Args:
            nin (int): Number of input features.
            nouts (list of int): List specifying the number of neurons in each layer.
            activation (str, optional): Activation function for hidden layers.
        """
        self.layers = []
        layers_sizes = [nin] + nouts
        for i in range(len(nouts)):
            activation_func = activation if i < len(nouts) - 1 else 'no_activation'
            self.layers.append(Layer(layers_sizes[i], layers_sizes[i+1], activation_func))

    def __call__(self, x):
        """
        Computes the MLP's output for a given input.

        Args:
            x (list of Value): Input values.

        Returns:
            Value or list of Value: Outputs from the final layer.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """
        Returns the parameters of the MLP.

        Returns:
            list of Value: All parameters from all layers.
        """
        return [p for layer in self.layers for p in layer.parameters()]