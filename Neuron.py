import math


class neuron:


    # The neurons output
    neuronOutput = 0.0

    # The neurons error and delta for back propagation
    error = 0.0
    delta = 0.0

    def __init__(self, weights, bias, classification):

        # Store the bias that was provided
        self.bias = bias

        self.classification = classification

        # Initialize the weights
        self.weights = weights

    def getNeuronOutput(self, inputs):

        # Sets the activation to the bias of the neuron
        activation = self.bias

        # Created the weighted sum from the neurons weights and the inputs provided
        for i in range(0, len(inputs) - 1):
            activation += self.weights[i] * inputs[i]

        # Apply the sigmoid transfer function to the activation
        output = 1.0 / (1.0 + math.exp(-activation))

        self.neuronOutput = output

        return output



