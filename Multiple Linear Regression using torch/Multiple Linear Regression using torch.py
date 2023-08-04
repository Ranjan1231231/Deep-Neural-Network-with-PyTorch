#Multiple Linear Regression using torch
import torch
from torch import nn

torch.manual_seed(1)

#prediction
#set the weight and bias

w = torch.tensor([[2.0], [3.0]], requires_grad=True)
b = torch.tensor([[1.0]], requires_grad=True)


#define prediction function
def forward(x):
    yhat=torch.mm(x,w)+b
    return yhat

# Calculate yhat

x = torch.tensor([[1.0, 2.0]])
yhat = forward(x)
print("The result: ", yhat)
# Sample tensor X

X = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])

# Make the prediction of X

yhat = forward(X)
print("The result: ", yhat)

#make a linear regression model using build in function
model=nn.Linear(2,1)
#make a prediction of x
yhat=model(x)
print("The result:",yhat)
#make a prediction of x
yhat=model(X)
print("the result:",yhat)
#built custum modelues
#create linear_regression class
class linear_regression(nn.Module):
    #constructor
    def __init__(self,input_size,output_Size):
        super(linear_regression, self).__init__()
        self.linear=nn.Linear(input_size,output_Size)
    #prediction function
    def forward( self,x ):
        yhat=self.linear(x)
        return  yhat
model=linear_regression(2,1)
print("the parameters:",list(model.parameters()))
print("the parameters:",model.state_dict())

#make a prediction of x
yhat=model(x)
print("The reult:",yhat)

#make a predictionof x
yhat=model(X)
print("the result :",yhat)

