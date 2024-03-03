import numpy as np

x=[[1,2,3,2.5],
   [2.0,5.0,-1.0,2.0],
   [-1.5, 2.8, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.10*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)
layer1.forward(x)
layer2.forward(layer1.output)
print(layer1.output)