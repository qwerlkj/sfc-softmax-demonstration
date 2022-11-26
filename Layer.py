import numpy as np


class Layer:

    def __init__(self, input_size=2, number_of_neurons=5, name="L1"):
        self.name = name
        self.neurons = np.empty(number_of_neurons)
        self.activation = np.empty(number_of_neurons)
        self.weights = np.random.rand(number_of_neurons, input_size)
        self.lr = 0.5
        self.l_input = np.zeros(input_size)
        self.delta = np.zeros(input_size)
        self.neuron_location = []
        self.expected = None

    def softmax(self, values):
        return np.exp(values) / np.sum(np.exp(self.neurons))

    def calculate_neuron(self, neuron_index: int, l_input: np.array):
        self.l_input = l_input
        out = np.sum(self.l_input * self.weights[neuron_index])
        self.neurons[neuron_index] = out

    def calculate_neurons(self, l_input: np.array):
        for i in range(self.neurons.shape[0]):
            self.calculate_neuron(i, l_input)

    def calculate_activation(self):
        for i in range(self.neurons.shape[0]):
            self.activation[i] = self.softmax(self.neurons[i])
        return self.activation

    def calculate_delta(self, delta=None, expected=None, weights=None):
        if delta is None:
            if expected is None:
                print("Should be defined to know what to learn")
            else:
                self.expected = expected
                self.delta = (expected - self.activation) * self.activation * (1 - self.activation)
        elif weights is not None:
            self.delta = np.sum(delta * weights.T) * self.activation * (1 - self.activation)
        else:
            print("If i have delta from next layer - i need weights too")

    def calculate_weights(self, x_input):
        multCoef = self.lr * self.delta
        for i in range(self.weights.shape[0]):
            print(">>>>>> ", self.weights[i], x_input, multCoef[i:i + 1])
            self.weights[i] += x_input * multCoef[i:i + 1]
