import torch
import torch.nn
from torch import Tensor
from torchtyping import TensorType

# Helpful functions:
# https://pytorch.org/docs/stable/generated/torch.reshape.html
# https://pytorch.org/docs/stable/generated/torch.mean.html
# https://pytorch.org/docs/stable/generated/torch.cat.html
# https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html

# Round your answers to 4 decimal places using torch.round(input_tensor, decimals = 4)
class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> Tensor: # a tensor to reshape
        M, N = to_reshape.shape
        done = torch.reshape(to_reshape, (M*N//2, 2))
        return torch.round(done, decimals=4)
        # torch.reshape() will be useful - check out the documentation
        pass

    def average(self, to_avg: TensorType[float]) -> Tensor: # a tensor to average column wise
        bog = torch.mean(to_avg, dim=0)
        return torch.round(bog, decimals=4)
        # torch.mean() will be useful - check out the documentation
        pass

    # catone -  the first tensor to concatenate, cattwo - the second tensor to concatenate
    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> Tensor:
        good = torch.cat((cat_one, cat_two), dim = 1)
        return good
        # torch.cat() will be useful - check out the documentation
        pass

    # prediction - the output tensor of a model, target - the true labels for the model inputs
    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> Tensor:
        joe = torch.nn.functional.mse_loss(prediction, target)
        return torch.round(joe, decimals=4)
        # torch.nn.functional.mse_loss() will be useful - check out the documentation
        pass

# Instantiate the Solution class
sol = Solution()

# Example for Reshape
to_reshape_input = torch.tensor([
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0]
], dtype=torch.float)

# Example for Average
to_avg_input = torch.tensor([
    [0.8088, 1.2614, -1.4371],
    [-0.0056, -0.2050, -0.7201]
], dtype=torch.float)

# Example for Concatenate
cat_one_input = torch.tensor([
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0]
], dtype=torch.float)

cat_two_input = torch.tensor([
    [1.0, 1.0],
    [1.0, 1.0]
], dtype=torch.float)

# Example for Get Loss
prediction_input = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0], dtype=torch.float)
target_input = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float)

# Test the functions with the provided input
reshape_output = sol.reshape(to_reshape_input)
average_output = sol.average(to_avg_input)
concatenate_output = sol.concatenate(cat_one_input, cat_two_input)
get_loss_output = sol.get_loss(prediction_input, target_input)

# Print the outputs
print("Reshape Output:")
print(reshape_output)
'''
Reshape Output:
tensor([[1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.]])

'''

print("\nAverage Output:")
print(average_output)
'''
Average Output:
tensor([ 0.4016,  0.5282, -1.0786])
'''

print("\nConcatenate Output:")
print(concatenate_output)
'''
Concatenate Output:
tensor([[1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.]])
'''

print("\nGet Loss Output:")
print(get_loss_output)
'''
Get Loss Output:
tensor(0.6000)
'''
