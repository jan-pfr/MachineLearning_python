import numpy as np
from sklearn import datasets
iris = datasets.load_iris()


# define the sigmoid function
def sigmoid(x, derivative=False):
    if (derivative == True):
        return sigmoid(x, derivative=False) * (1 - sigmoid(x, derivative=False))
    else:
        return 1 / (1 + np.exp(-x))


# choose a random seed for reproducible results
np.random.seed(1)

# learning rate
alpha = .1

# number of nodes in the hidden layer
num_hidden = 10

# inputs
X = iris.data[:, :2]  # 4 attributes are available, only the first two are used.
print(X)
# outputs
# x.T is the transpose of x, making this a column vector
y = iris.target
np.transpose(y)
print(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=r)

# initialize weights randomly with mean 0 and range [-1, 1]
# the +1 in the 1st dimension of the weight matrices is for the bias weight
hidden_weights = 2 * np.random.random((X.shape[1] + 1, num_hidden)) - 1
output_weights = 2 * np.random.random((num_hidden + 1, y.shape[0])) - 1

# number of iterations of gradient descent
num_iterations = 10000

# for each iteration of gradient descent
for i in range(num_iterations):
    # forward phase
    # np.hstack((np.ones(...), X) adds a fixed input of 1 for the bias weight
    input_layer_outputs = np.hstack((np.ones((X.shape[0], 1)), X))
    hidden_layer_outputs = np.hstack((np.ones((X.shape[0], 1)), sigmoid(np.dot(input_layer_outputs, hidden_weights))))
    output_layer_outputs = np.dot(hidden_layer_outputs, output_weights)

    # backward phase
    # output layer error term
    output_error = output_layer_outputs - y
    # hidden layer error term
    # [:, 1:] removes the bias term from the backpropagation
    hidden_error = hidden_layer_outputs[:, 1:] * (1 - hidden_layer_outputs[:, 1:]) * np.dot(output_error,
                                                                                            output_weights.T[:, 1:])

    # partial derivatives
    hidden_pd = input_layer_outputs[:, :, np.newaxis] * hidden_error[:, np.newaxis, :]
    output_pd = hidden_layer_outputs[:, :, np.newaxis] * output_error[:, np.newaxis, :]

    # average for total gradients
    total_hidden_gradient = np.average(hidden_pd, axis=0)
    total_output_gradient = np.average(output_pd, axis=0)

    # update weights
    hidden_weights += - alpha * total_hidden_gradient
    output_weights += - alpha * total_output_gradient

# print the final outputs of the neural network on the inputs X
print("Output After Training: \n{}".format(output_layer_outputs))
