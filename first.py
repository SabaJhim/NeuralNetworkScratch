import sys
import numpy as np 
import matplotlib

print("Python: ",sys.version)


#Just one neuron
input=[1,2,3]
weights=[0.2,0.8,-0.5]
bias=2

output=input[0]*weights[0]+input[1]*weights[1]+input[2]*weights[2]+bias
print(output)

# 3 neurons -> output layer

input=[1,2,3,2.5]
weight1=[0.2,0.8,-0.5,1.0]
weight2=[0.5,-0.91,0.26,-0.5]
weight3=[-0.26,-0.27,0.17,0.87]

bias1=2
bias2=3
bias3=0.5
output=[input[0]*weight1[0]+input[1]*weight1[1]+input[2]*weight1[2]+input[3]*weight1[3]+bias1,
        input[0]*weight2[0]+input[1]*weight2[1]+input[2]*weight2[2]+input[3]*weight2[3]+bias2,
        input[0]*weight3[0]+input[1]*weight3[1]+input[2]*weight3[2]+input[3]*weight3[3]+bias3
]

print(output)

# Simple numpy

weights=[[0.2,0.8,-0.5,1.0],
        [0.5,-0.91,0.26,-0.5],
        [-0.26,-0.27,0.17,0.87]]

biases=[2,3,0.5]

output_simple=np.dot(weights,input)+biases
print(output_simple)

#Batch input
inputs=[[1.0,2.0,3.0,2.5],
        [2.0,5.0,-1.0,2.0],
        [-1.5,2.7,3.3,-0.8]

]

output_batch=np.dot(inputs,np.array(weights).T)+biases
print(output_batch)

#Layer adding 

inputs=[[1.0,2.0,3.0,2.5],
        [2.0,5.0,-1.0,2.0],
        [-1.5,2.7,3.3,-0.8]]

weights1=[[0.2,0.8,-0.5,1.0],
        [0.5,-0.91,0.26,-0.5],
        [-0.26,-0.27,0.17,0.87]]

biases1=[2,3,0.5]

weights2=[[0.1,-0.14,0.5],
        [-0.5,0.12,-0.33],
        [-0.44,0.73,-0.13]]
  
biases2=[-1.0,2.0,-0.5]

layer1_output=np.dot(inputs,np.array(weights).T)+biases1
layer2_output=np.dot(layer1_output,np.array(weights2).T)+biases2

print(layer2_output)