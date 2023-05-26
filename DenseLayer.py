from numpy.random import normal
from numpy import zeros, dot, array, mean


class DenseLayer:
    def __init__(self, input_neurons=2, output_neurons=2):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.weights = normal(0, 1, size=(self.output_neurons, self.input_neurons))
        self.bias = zeros(self.output_neurons)
        self.previous_layer = None
        self.next_layer = None

    def call(self, X):
        if self.previous_layer is None:
            return array(
                [dot(X, self.weights[i]) + self.bias[i]
                 for i in range(self.output_neurons)]
            )
        return array(
            [dot(self.previous_layer.call(X), self.weights[i]) + self.bias[i]
             for i in range(self.output_neurons)]
        )

    def get_weight_d(self, i, j):
        if self.next_layer is None: return 1
        return self.weights[i][j] *\
            mean([self.next_layer.get_weight_d(j, u) for u in range(self.output_neurons)])




