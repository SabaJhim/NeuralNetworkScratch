import sys
import numpy as np
import matplotlib
import nnfs
from nnfs.datasets import spiral_data



nnfs.init()
np.random.seed(0)



X=[[1.0,2.0,3.0,2.5],
        [2.0,5.0,-1.0,2.0],
        [-1.5,2.7,3.3,-0.8]

]



class Activation_ReLU:
        def forward(self,inputs):
                self.output=np.maximum(0,inputs)

class Layer_dense : 
        def __init__(self,n_inputs,n_neurons):
                self.weights=0.1*np.random.randn(n_inputs,n_neurons)
                self.biases=np.zeros((1,n_neurons))
        def forward(self,inputs):
                self.output=np.dot(np.array(inputs),np.array(self.weights))+np.array(self.biases)


layer1=Layer_dense(4,5)
layer2=Layer_dense(5,2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)


#Adding Rectified Linear Unit

X,y=spiral_data(100,3)
layerA=Layer_dense(2,5)
layerA.forward(X)

activationA=Activation_ReLU()
activationA.forward(layerA.output)

print(activationA.output)