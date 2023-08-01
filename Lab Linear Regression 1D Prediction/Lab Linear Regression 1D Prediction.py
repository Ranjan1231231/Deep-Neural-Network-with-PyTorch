#Traning one parameter
import numpy as np
import matplotlib.pyplot as plt
import torch


#The class for plotting
class plot_diagram():
    #constructor
    def __init__(self,X,Y,w,stop,go=False):
        start=w.data
        self.error=[]
        self.parameter=[]
        print ( type ( X.numpy ( ) ) )
        self.X = X.numpy ( )

        self.Y = Y.numpy ( )
        self.parameter_values = torch.arange ( start , stop )
        self.Loss_function = [ criterion ( forward ( X ) , Y ) for w.data in self.parameter_values ]
        w.data = start

        # Executor

    def __call__ ( self , Yhat , w , error , n ) :
        self.error.append ( error )
        self.parameter.append ( w.data )
        plt.subplot ( 212 )
        plt.plot ( self.X , Yhat.detach ( ).numpy ( ) )
        plt.plot ( self.X , self.Y , 'ro' )
        plt.xlabel ( "A" )
        plt.ylim ( -20 , 20 )
        plt.subplot ( 211 )
        plt.title ( "Data Space (top) Estimated Line (bottom) Iteration " + str ( n ) )
        plt.show()
        # Convert lists to PyTorch tensors
        parameter_values_tensor = torch.tensor ( self.parameter_values )
        loss_function_tensor = torch.tensor ( self.Loss_function )
        # Plot using the tensors
        plt.plot ( parameter_values_tensor.numpy ( ) , loss_function_tensor.numpy ( ) )
        plt.plot ( self.parameter , self.error , 'ro' )
        plt.xlabel ( "B" )
        plt.figure ( )

        # Destructor

    def __del__ ( self ) :
        plt.close ( 'all' )


#Create the f(X) with a slope of -3
X=torch.arange(-3,3,0.10).view(-1,1)
f=-3*X

#plot the line with blue
# plt.plot(X.numpy(),f.numpy(),label='f')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.legend()
# plt.show()
#
Y = f + 0.1 * torch.randn(X.size())
#creating the model and cost function(total loss)
#create forward function for prediction
def forward(x):
    return w*x
#create the MSE for evaluate the result
def criterion(yhat,y):
    return torch.mean((yhat-y)**2)
#create learning rate and an emplty list to record the loss for each iteration
lr=0.1
LOSS=[]

w=torch.tensor(-10.0,requires_grad = True)
gradient_plot=plot_diagram(X,Y,w,stop=5)

#train the model

#define a function for train the model
def train_model(iter):
    for epoch in range(iter):
        #make the preidection as we learned in the last lab
        Yhat=forward(X)
        #calculate the iteration
        loss=criterion(Yhat,Y)
        # plot the diagram for us to have a better idea
        gradient_plot ( Yhat , w , loss.item ( ) , epoch )
        # store the loss into list
        LOSS.append ( loss.item ( ) )
        # backward pass: compute gradient of the loss with respect to all the learnable parameters
        loss.backward ( )
        # updata parameters
        w.data = w.data - lr * w.grad.data
        # zero the gradients before running the backward pass
        w.grad.data.zero_ ( )


# train_model(4)


#plot the loss for each iteration
# plt.plot(LOSS)
# plt.tight_layout()
# plt.xlabel("Epoch/Iterations")
# plt.ylabel("Cost")
# plt.show()