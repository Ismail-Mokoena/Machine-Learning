import numpy as np

#rectifier linear activation function
def relu_activation(input) -> np.ndarray:
    output = np.maximum(0,input)
    return output

#softmax activation function
def softmax(layer_outputs) -> np.ndarray:
    #layer_outputs-np.max protects against overflow
    e_val = np.exp(layer_outputs-np.max(layer_outputs, axis=1,keepdims=True))
    normalization_val = e_val/np.sum(e_val, axis=1, keepdims=True)
    return normalization_val