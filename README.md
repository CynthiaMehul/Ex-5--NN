### Cynthia Mehul  
### 212223240020  
### EX. NO.5  
### DATE:  

# Implementation of XOR using RBF

### Aim:
To implement a XOR gate classification using Radial Basis Function Neural Network.

### Theory:

Exclusive OR (XOR) is a logical operation that outputs true when the inputs differ. For the XOR gate, the truth table is as follows:

XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in the figure below:

The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.

A Radial Basis Function Network (RBFN) is a particular type of neural network. The RBFN approach is more intuitive than MLP. An RBFN performs classification by measuring the input’s similarity to examples from the training set. Each RBFN neuron stores a “prototype,” which is just one of the examples from the training set. When we want to classify a new input, each neuron computes the Euclidean distance between the input and its prototype. Thus, if the input more closely resembles the class A prototypes than the class B prototypes, it is classified as class A, else class B.

A neural network with an input layer, one hidden layer with Radial Basis Function, and a single-node output layer (as shown in the figure below) will be able to classify the binary data according to the XOR output.

### Algorithm:

1. Initialize the input vector for your bit binary data  
2. Initialize the centers for two hidden neurons in the hidden layer  
3. Define the nonlinear function for the hidden neurons using Gaussian RBF  
4. Initialize the weights for the hidden neurons  
5. Determine the output function as:  
   `Y = W1 * φ1 + W2 * φ2`  
6. Test the network for accuracy  
7. Plot the input space and hidden space of RBF NN for XOR classification  

### Program:

```python
import numpy as np
import matplotlib.pyplot as plt

def gaussian_rbf(x, landmark, gamma=1):
    return np.exp(-gamma * np.linalg.norm(np.array(x) - np.array(landmark))**2)

x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
ys = [0, 1, 1, 0]
X1, X2 = x1, x2

mu1 = [0, 0]
mu2 = [1, 1]

from_1 = [gaussian_rbf(i, mu1) for i in zip(X1, X2)]
from_2 = [gaussian_rbf(i, mu2) for i in zip(X1, X2)]

plt.figure(figsize=(13, 5))

# Original Input Space
plt.subplot(1, 2, 1)
plt.scatter([x1[0], x1[3]], [x2[0], x2[3]], label="Class_0")
plt.scatter([x1[1], x1[2]], [x2[1], x2[2]], label="Class_1")
plt.xlabel("$X1$", fontsize=15)
plt.ylabel("$X2$", fontsize=15)
plt.title("XOR: Linearly Inseparable", fontsize=15)
plt.legend()

# Transformed Feature Space
plt.subplot(1, 2, 2)
plt.scatter(from_1[0], from_2[0], label="Class_0")
plt.scatter(from_1[1], from_2[1], label="Class_1")
plt.scatter(from_1[2], from_2[2], label="Class_1")
plt.scatter(from_1[3], from_2[3], label="Class_0")
plt.plot([0, 0.95], [0.95, 0], "k--")
plt.annotate("Separating hyperplane", xy=(0.4, 0.55), xytext=(0.55, 0.66),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.xlabel(f"$\\mu_1$: {mu1}", fontsize=15)
plt.ylabel(f"$\\mu_2$: {mu2}", fontsize=15)
plt.title("Transformed Inputs: Linearly Separable", fontsize=15)
plt.legend()
plt.show()
```
### Output:

![image](https://github.com/user-attachments/assets/a50c75e5-29a5-4eb2-920a-1a86cc4df45a)

![image](https://github.com/user-attachments/assets/73a58d62-0fad-43d7-ab97-fd6f97e1a89d)

### Result:
Thus, a Radial Basis Function Neural Network is implemented to classify XOR data.
