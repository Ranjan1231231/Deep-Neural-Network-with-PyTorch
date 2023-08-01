import torch
from torch.nn import Linear
from torch import nn
# Define w = 2 and b = -1 for y = wx + b
w = torch.tensor(2.0, requires_grad = True)
b = torch.tensor(-1.0, requires_grad = True)

# Function forward(x) for prediction
def forward(x):
    yhat = w * x + b
    return yhat

# Predict y = 2x - 1 at x = 1
x = torch.tensor([[1.0]])
yhat = forward(x)
# print("The prediction: ", yhat)

# Create x Tensor and check the shape of x tensor
x = torch.tensor([[1.0], [2.0]])
# print("The shape of x: ", x.shape)


## Make the prediction of y = 2x - 1 at x = [1, 2]
yhat = forward(x)
# print("The prediction: ", yhat)


x = torch.tensor([[1.0], [2.0], [3.0]])
yhat = forward(x)
# print("The prediction: ", yhat)

#set random seed
torch.manual_seed(1)


# Create Linear Regression Model, and print out the parameters
lr = Linear(in_features=1, out_features=1, bias=True)
# print("Parameters w and b: ", list(lr.parameters()))

# print("Python dictionary: ",lr.state_dict())
# print("keys: ",lr.state_dict().keys())
# print("values: ",lr.state_dict().values())
# print("weight:",lr.weight)
# print("bias:",lr.bias)

# Make the prediction at x = [[1.0]]
x = torch.tensor([[1.0]])
yhat = lr(x)
# print("The prediction: ", yhat)

# Create the prediction using linear model

x = torch.tensor([[1.0], [2.0]])
yhat = lr(x)
# print("The prediction: ", yhat)


#practice
x = torch.tensor([[1.0],[2.0],[3.0]])
yhat = lr(x)
# print("The prediction: ", yhat)


#BUILD CUSTOM MODULES
# Customize Linear Regression Class

class LR ( nn.Module ) :
    # Constructor
    def __init__ ( self , input_size , output_size ) :
        # Inherit from parent
        super ( LR , self ).__init__ ( )
        self.linear = nn.Linear ( input_size , output_size )

    # Prediction function
    def forward ( self , x ) :
        out = self.linear ( x )
        return out
# Create the linear regression model. Print out the parameters.
lr = LR(1, 1)
# print("The parameters: ", list(lr.parameters()))
# print("Linear model: ", lr.linear)

# Try our customize linear regression model with single input
x = torch.tensor([[1.0]])
yhat = lr(x)
# print("The prediction: ", yhat)


#Try our customize linear regression model with multiple input
x = torch.tensor([[1.0], [2.0]])
yhat = lr(x)
# print("The prediction: ", yhat)

# print("Python dictionary: ", lr.state_dict())
# print("keys: ",lr.state_dict().keys())
# print("values: ",lr.state_dict().values())

