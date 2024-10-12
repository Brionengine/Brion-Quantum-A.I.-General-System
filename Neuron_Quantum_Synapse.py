import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, num_inputs, activation='relu'):
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand()
        self.activation = relu if activation == 'relu' else sigmoid

    def forward(self, inputs):
        weighted_sum = np.dot(self.weights, inputs) + self.bias
        return self.activation(weighted_sum)

class NeuronLayer:
    def __init__(self, num_neurons, num_inputs_per_neuron, activation='relu'):
        self.neurons = [Neuron(num_inputs_per_neuron, activation) for _ in range(num_neurons)]

    def forward(self, inputs):
        return np.array([neuron.forward(inputs) for neuron in self.neurons])

if __name__ == "__main__":
    inputs = np.array([0.5, -0.2, 0.1])
    neuron_layer = NeuronLayer(num_neurons=3, num_inputs_per_neuron=3, activation='relu')
    layer_output = neuron_layer.forward(inputs)
    print(f"Layer output (ReLU): {layer_output}")
    
    neuron_layer_sigmoid = NeuronLayer(num_neurons=3, num_inputs_per_neuron=3, activation='sigmoid')
    layer_output_sigmoid = neuron_layer_sigmoid.forward(inputs)
    print(f"Layer output (Sigmoid): {layer_output_sigmoid}")
