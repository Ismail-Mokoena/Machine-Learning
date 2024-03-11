import numpy as np
from dataclasses import dataclass
from Activation import *
from Loss import loss

x=[ [1,2,3,2.5],
   [2.0,5.0,-1.0,2.0],
   [-1.5, 2.8, 3.3, -0.8]]

y_true = np.array([[1,0,0,0],
          [0,1,0,0],
          [0,1,0,0]])

@dataclass
class LayerDense:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.10*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = LayerDense(4,3)
layer2 = LayerDense(3,4)

layer1.forward(x)
act1 = relu_activation(layer1.output)

layer2.forward(act1)

prediction = softmax(layer2.output)

print("Loss = ",loss(prediction, y_true))
