import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#DERIVATIVES

#basic derivation
#creating a tensor x
x=torch.tensor(2.0,requires_grad = True)
# print("The tensor x: ",x)
#creating a tensor y according to the function y=x**2
y=x**3
# print("The result of y=x^2",y)
#taking the derivative and trying to print out the derivative at value x=2
y.backward()#backward function converts the derivative
# print("The derivative at x=2:",x.grad)
#
#
# print('data:',x.data)
# print('grad_fn:',x.grad_fn)
# print('grad:',x.grad)
# print('is_leaf:',x.is_leaf)
# print('reaquires_grad:',x.requires_grad)


# print('data:',y.data)
# print('grad_fn:',y.grad_fn)
# print('grad:',y.grad)
# print("is_leaf:",y.is_leaf)
# print("requires_grad:",y.requires_grad)

#calculating the derivative for a more complicated function
x=torch.tensor(2.0,requires_grad = True)
y=x**2+2*x+1
# print("The result of y=x^2+2x+1:",y)
y.backward()
# print("the derivative at x=2:",x.grad)


#Example 2
x=torch.tensor(1.0,requires_grad = True)
y=2*x**3+x
y.backward()
# print('The derivative at x=1:',x.grad)

#CREATING OUR OWN CUSTOM AUTOGRAD FUNCTION BY SUBCLASSING TORCH.AUTOGRAD FUNCTION AND IMPLIMENTING THE FORWARD AND BACKWARD PASSES WHICH OPERATE ON TENSORS
class SQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx,i):
        """
               In the forward pass we receive a Tensor containing the input and return
               a Tensor containing the output. ctx is a context object that can be used
               to stash information for backward computation. You can cache arbitrary
               objects for use in the backward pass using the ctx.save_for_backward method.
               """
        result=i**2
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx,grad_output):
        """
                In the backward pass we receive a Tensor containing the gradient of the loss
                with respect to the output, and we need to compute the gradient of the loss
                with respect to the input.
                """
        i, =ctx.saved_tensors
        grad_output=2*i
        return grad_output

#Applying the function
x=torch.tensor(2.0,requires_grad = True)
sq=SQ.apply
y=sq(x)
# print(y)
# print(y.grad_fn)
y.backward()
# print(x.grad)


#PARTIAL DERIVATIVES
#CALCULATE F(U,V)=V*U+U^2 AT U=1,V=2
u=torch.tensor(1.0,requires_grad = True)
v=torch.tensor(2.0,requires_grad = True)
f=u*v+u**2
# print("The result of v*u+u^2:",f)
f.backward()

#calculation of derivative with respect to u
# print("The partial derivative with respect to u:",u.grad)
#calculation of   derivative with respective to v
# print("The partial derivatice with respect to v:",v.grad)

#calculate the derivative with multiple values
x=torch.linspace(-10,10,10,requires_grad=True)
Y=x**2
y=torch.sum(x**2)

#take the derivative with respect to multiple value . plot out the function and its derivative
y.backward()
# plt.plot(x.detach().numpy(),Y.detach().numpy(),label='function')
# plt.plot(x.detach().numpy(),x.grad.detach().numpy(),label='derivative')
# plt.xlabel('x')
# plt.legend()
# plt.show()


#take the derivative of relu with respect to multiple value , plot out the function ad its derivative
x=torch.linspace(-10,10,1000,requires_grad=True)
Y=torch.relu(x)
y=Y.sum()
y.backward()
# plt.plot(x.detach().numpy(), Y.detach().numpy(), label = 'function')
# plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label = 'derivative')
# plt.xlabel('x')
# plt.legend()
# plt.show()

# print(y.grad_fn)
