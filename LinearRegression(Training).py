import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int,
                       X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(self, X: NDArray[np.float64], Y: NDArray[np.float64], num_iterations: int,
                    initial_weights: NDArray[np.float64]
                    ) -> NDArray[np.float64]:
        for _ in range(num_iterations):
            bog = self.get_model_prediction(X, initial_weights)

            # get derivative for each weights
            d1 = self.get_derivative(bog, Y, len(X), X, 0)
            d2 = self.get_derivative(bog, Y, len(X), X, 1)
            d3 = self.get_derivative(bog, Y, len(X), X, 2)

            # Gradient Descent on each weight
            initial_weights[0] = initial_weights[0] - d1 * self.learning_rate
            initial_weights[1] = initial_weights[1] - d2 * self.learning_rate
            initial_weights[2] = initial_weights[2] - d3 * self.learning_rate

        return np.round(initial_weights, 5)

        # you will need to call get_derivative() for each weight
        # and update each one separately based on the learning rate!
        # return np.round(your_answer, 5)

        pass

'''
Input:
X=[[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]
Y=[6.0, 3.0]
num_iterations=10
initial_weights=[0.2, 0.1, 0.6]

Output:
[0.50678, 0.59057, 1.27435]

The initial weights are [0.2, 0.1, 0.6].
The algorithm iteratively updates the weights using gradient descent to 
minimize the mean squared error between the predicted and actual values.

The goal is to find weights that result in a model that best fits the given input 
data (X) to produce predictions close to the actual target values (Y).
'''

X = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
Y = np.array([6.0, 3.0])
num_iterations = 10
initial_weights = np.array([0.2, 0.1, 0.6])

solution = Solution()
result = solution.train_model(X, Y, num_iterations, initial_weights)
print(result)
# [0.50678 0.59057 1.27435]
