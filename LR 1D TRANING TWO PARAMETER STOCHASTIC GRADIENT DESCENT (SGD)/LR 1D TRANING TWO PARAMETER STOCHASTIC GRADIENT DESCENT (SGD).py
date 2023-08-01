#Linear regression 1D: Training Two Parameter Stochastic Gradient Descent (SGD)
import  torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from torch.utils.data import Dataset,DataLoader
#the class for plot the diagram
class plot_error_surfaces ( object ) :

    # Constructor
    def __init__ ( self , w_range , b_range , X , Y , n_samples = 30 , go = True ) :
        W = np.linspace ( -w_range , w_range , n_samples )
        B = np.linspace ( -b_range , b_range , n_samples )
        w , b = np.meshgrid ( W , B )
        Z = np.zeros ( (30 , 30) )
        count1 = 0
        self.y = Y.numpy ( )
        self.x = X.numpy ( )
        for w1 , b1 in zip ( w , b ) :
            count2 = 0
            for w2 , b2 in zip ( w1 , b1 ) :
                Z [ count1 , count2 ] = np.mean ( (self.y - w2 * self.x + b2) ** 2 )
                count2 += 1
            count1 += 1
        self.Z = Z
        self.w = w
        self.b = b
        self.W = [ ]
        self.B = [ ]
        self.LOSS = [ ]
        self.n = 0
        if go == True :
            plt.figure ( )
            plt.figure ( figsize = (7.5 , 5) )
            plt.axes ( projection = '3d' ).plot_surface ( self.w , self.b , self.Z , rstride = 1 , cstride = 1 ,
                                                          cmap = 'viridis' , edgecolor = 'none' )
            plt.title ( 'Loss Surface' )
            plt.xlabel ( 'w' )
            plt.ylabel ( 'b' )
            plt.show ( )
            plt.figure ( )
            plt.title ( 'Loss Surface Contour' )
            plt.xlabel ( 'w' )
            plt.ylabel ( 'b' )
            plt.contour ( self.w , self.b , self.Z )
            plt.show ( )

    # Setter
    def set_para_loss ( self , W , B , loss ) :
        self.n = self.n + 1
        self.W.append ( W )
        self.B.append ( B )
        self.LOSS.append ( loss )

    # Plot diagram
    def final_plot ( self ) :
        ax = plt.axes ( projection = '3d' )
        ax.plot_wireframe ( self.w , self.b , self.Z )
        ax.scatter ( self.W , self.B , self.LOSS , c = 'r' , marker = 'x' , s = 200 , alpha = 1 )
        plt.figure ( )
        plt.contour ( self.w , self.b , self.Z )
        plt.scatter ( self.W , self.B , c = 'r' , marker = 'x' )
        plt.xlabel ( 'w' )
        plt.ylabel ( 'b' )
        plt.show ( )

    # Plot diagram
    def plot_ps ( self ) :
        plt.subplot ( 121 )
        plt.ylim
        plt.plot ( self.x , self.y , 'ro' , label = "training points" )
        plt.plot ( self.x , self.W [ -1 ] * self.x + self.B [ -1 ] , label = "estimated line" )
        plt.xlabel ( 'x' )
        plt.ylabel ( 'y' )
        plt.ylim ( (-10 , 15) )
        plt.title ( 'Data Space Iteration: ' + str ( self.n ) )
        plt.subplot ( 122 )
        plt.contour ( self.w , self.b , self.Z )
        plt.scatter ( self.W , self.B , c = 'r' , marker = 'x' )
        plt.title ( 'Loss Surface Contour Iteration' + str ( self.n ) )
        plt.xlabel ( 'w' )
        plt.ylabel ( 'b' )
        plt.show ( )

#making the data
torch.manual_seed(1)
X=torch.arange(-3,3,0.1).view(-1,1)
f=1*X-1
Y=f+0.1*torch.randn(X.size())
#setup the  actual data and simulated data
# plt.plot(X.numpy(),Y.numpy(),'rx',label='y')
# plt.plot(X.numpy(),f.numpy(),label='f')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()


#define the forward dunction
def forward(x):
    return w*x+b
#define the mseloss function
def criterion(yhat,y):
    return  torch.mean((yhat-y)**2)
#create plot-surface for viewing the data
# get_surface = plot_error_surfaces(15, 13, X, Y, 30)

#TRANING THE MODEL, BATCH GRADIENT DESENT
# Define the parameters w, b for y = wx + b
w = torch.tensor(-15.0, requires_grad = True)
b = torch.tensor(-10.0, requires_grad = True)

# Define learning rate and create an empty list for containing the loss for each iteration.
lr = 0.1
LOSS_BGD = []

# Create plot_error_surfaces for viewing the data

get_surface = plot_error_surfaces(15, 13, X, Y, 30, go = False)
# The function for training the model

def train_model ( iter ) :
    # Loop
    for epoch in range ( iter ) :
        # make a prediction
        Yhat = forward ( X )

        # calculate the loss
        loss = criterion ( Yhat , Y )

        # Section for plotting
        get_surface.set_para_loss ( w.data.tolist ( ) , b.data.tolist ( ) , loss.tolist ( ) )
        get_surface.plot_ps ( )

        # store the loss in the list LOSS_BGD
        LOSS_BGD.append ( loss )

        # backward pass: compute gradient of the loss with respect to all the learnable parameters
        loss.backward ( )

        # update parameters slope and bias
        w.data = w.data - lr * w.grad.data
        b.data = b.data - lr * b.grad.data

        # zero the gradients before running the backward pass
        w.grad.data.zero_ ( )
        b.grad.data.zero_ ( )

#TRAIN THE MODEL WITH 10 ITERATIONS
# train_model(10)

#dataset class
class Data(Dataset):
    #constructor
    def __init__(self):
        self.x=torch.arange(-3,3,0.1).view(-1,1)
        self.y=1*self.x-1
        self.len=self.x.shape[0]
    #getter
    def __getitem__(self, index):
        return self.x[index],self.y[index]
    #return the length
    def __len__(self):
        return  self.len
#create the dataset and check the length
dataset=Data()
# print("the length of dataset:",len(dataset))


#print the first ppoint
x,y=dataset[0]
# print("(", x, ", ", y, ")")
#similarly print the first three point
x, y = dataset[0:3]
# print("The first 3 x: ", x)
# print("The first 3 y: ", y)

#create plot_error surface for viewing the data
get_surface=plot_error_surfaces(15,13,X,Y,30,go = False)

#create dataloader
trainloader = DataLoader(dataset = dataset, batch_size = 1)

#The function for traning the model

w = torch.tensor ( -15.0 , requires_grad = True )
b = torch.tensor ( -10.0 , requires_grad = True )
LOSS_Loader = [ ]


def train_model_DataLoader ( epochs ) :
    # Loop
    for epoch in range ( epochs ) :

        # SGD is an approximation of out true total loss/cost, in this line of code we calculate our true loss/cost and store it
        Yhat = forward ( X )

        # store the loss
        LOSS_Loader.append ( criterion ( Yhat , Y ).tolist ( ) )

        for x , y in trainloader :
            # make a prediction
            yhat = forward ( x )

            # calculate the loss
            loss = criterion ( yhat , y )

            # Section for plotting
            get_surface.set_para_loss ( w.data.tolist ( ) , b.data.tolist ( ) , loss.tolist ( ) )

            # Backward pass: compute gradient of the loss with respect to all the learnable parameters
            loss.backward ( )

            # Updata parameters slope
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data

            # Clear gradients
            w.grad.data.zero_ ( )
            b.grad.data.zero_ ( )

        # plot surface and data space after each epoch
        get_surface.plot_ps ( )
# Run 10 iterations
train_model_DataLoader(10)
# Plot the LOSS_BGD and LOSS_Loader
plt.plot(LOSS_BGD,label="Batch Gradient Descent")
plt.plot(LOSS_Loader,label="Stochastic Gradient Descent with DataLoader")
plt.xlabel('epoch')
plt.ylabel('Cost/ total loss')
plt.legend()
plt.show()

