from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# a relu activation function returns
# 0 for negative numbers and the 
# number itself for positive function
def relu(input):
    return max(0,input)

# this is the main function for 
# neural network it a simple single
# layer neural network with only one
# hidden layer  
def predict(input, weights):
    node_0_input=(input*weights[0]).sum()
    node_1_input=(input*weights[1]).sum()

    node_0_output=relu(node_0_input)
    node_1_output=relu(node_1_input)

    hidden_layer_output=np.array([node_0_output,node_0_input])
    output=(hidden_layer_output*weights[2]).sum()

    return output

# random definition for weights, input
#  and output
weights=[ np.array([1,1]), np.array([1,0]), np.array([2,1])]
input=np.array([0,3])
output=6

# learning rate, i.e the kind  of the 
# dx  in loss function
learning_rate = 0.01 

error_history=[]

# the loop for learning  
# here error is just the absolute error 
# the weights are update in a similar way to
# gradient descent to minimise the error 

for i in range(30):
    predicted_output=predict( input, weights )
    error=abs( output-predicted_output )
    slope=2 * input * error 
    weights=weights - learning_rate * slope
    error_history.append(error)

plt.plot(error_history)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.show()