#LINEAR REGRESSION 1D - TRANING TWO PARAMETER MINI BATCH GRADIENT DECENT
from torch.utils.data import Dataset,DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
#the class for plotting the diagram
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
        plt.ylim ( )
        plt.plot ( self.x , self.y , 'ro' , label = "training points" )
        plt.plot ( self.x , self.W [ -1 ] * self.x + self.B [ -1 ] , label = "estimated line" )
        plt.xlabel ( 'x' )
        plt.ylabel ( 'y' )
        plt.title ( 'Data Space Iteration: ' + str ( self.n ) )
        plt.subplot ( 122 )
        plt.contour ( self.w , self.b , self.Z )
        plt.scatter ( self.W , self.B , c = 'r' , marker = 'x' )
        plt.title ( 'Loss Surface Contour' )
        plt.xlabel ( 'w' )
        plt.ylabel ( 'b' )
        plt.show ( )

#MAKING SOME DATA
#import pytorch library
torch.manual_seed(1)
#generate the data with noise and the line
X=torch.arange(-3,3,0.1).view(-1,1)
f=1*X-1
Y=f+0.1*torch.randn(X.size())
#plotting the results
# plt.plot(X.numpy(),Y.numpy(),'rx',label='y')
# plt.plot(X.numpy(),f.numpy(),label='f')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()


#create the model and cost function(total loss)
#define the prediction function
def forward(x):
    return  w*x+b
#define the cost function
def criterion(yhat,y):
    return torch.mean((yhat-y)**2)
#create a plot_error_surfaces object
get_surface=plot_error_surfaces(15,13,X,Y,30)


#TRAIN THE MODEL:BATCH GRADIENT DESCENT
#define the function for training model
w=torch.tensor(-15.0,requires_grad = True)
b=torch.tensor(-10.0,requires_grad = True)
lr=0.1
LOSS_BGD=[]
def train_model_BGD(epochs):
    for epoch in range(epochs):
        Yhat=forward(X)
        loss=criterion(Yhat,Y)
        LOSS_BGD.append(loss)
        get_surface.set_para_loss(w.data.tolist(),b.data.tolist(),loss.tolist())
        get_surface.plot_ps()
        loss.backward()
        w.data=w.data-lr*w.grad.data
        b.data=b.data-lr*b.grad.data
        w.grad.data.zero_()
        b.grad.data.zero_()

#run  train_model_BGD with 10 iteration
# train_model_BGD(10)
#stochastic gradient descent(sgd)with  dataset dataloader
#create a plot_error_surfaces object
get_surface=plot_error_surfaces(15,13,X,Y,30,go=False)

#create class data
# Create class Data

class Data ( Dataset ) :

    # Constructor
    def __init__ ( self ) :
        self.x = torch.arange ( -3 , 3 , 0.1 ).view ( -1 , 1 )
        self.y = 1 * X - 1
        self.len = self.x.shape [ 0 ]

    # Getter
    def __getitem__ ( self , index ) :
        return self.x [ index ] , self.y [ index ]

    # Get length
    def __len__ ( self ) :
        return self.len


#create data object and dataloader object
dataset=Data()
trainloader=DataLoader(dataset=dataset,batch_size = 1)
#define train_mode_SGD function
w=torch.tensor(-15.0,requires_grad = True)
b=torch.tensor(-10.0,requires_grad = True)

LOSS_SGD=[]
lr=0.1
def train_model_SGD(epochs):
    for epoch in range(epochs):
        Yhat = forward(X)
        get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), criterion(Yhat, Y).tolist())
        get_surface.plot_ps()
        LOSS_SGD.append(criterion(forward(X), Y).tolist())
        for x, y in trainloader:
            yhat = forward(x)
            loss = criterion(yhat, y)
            get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
            loss.backward()
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data
            w.grad.data.zero_()
            b.grad.data.zero_()
        get_surface.plot_ps()

#run train_model_surface object
train_model_SGD(10)


# Create a plot_error_surfaces object.

get_surface = plot_error_surfaces(15, 13, X, Y, 30, go = False)

# Create DataLoader object and Data object

dataset = Data()
trainloader = DataLoader(dataset = dataset, batch_size = 5)

# Define train_model_Mini5 function

w = torch.tensor(-15.0, requires_grad = True)
b = torch.tensor(-10.0, requires_grad = True)
LOSS_MINI5 = []
lr = 0.1

def train_model_Mini5(epochs):
    for epoch in range(epochs):
        Yhat = forward(X)
        get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), criterion(Yhat, Y).tolist())
        get_surface.plot_ps()
        LOSS_MINI5.append(criterion(forward(X), Y).tolist())
        for x, y in trainloader:
            yhat = forward(x)
            loss = criterion(yhat, y)
            get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
            loss.backward()
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data
            w.grad.data.zero_()
            b.grad.data.zero_()
#RUN TRAIN MODEL_MINI5 WITH 10 ITERATION
# Run train_model_Mini5 with 10 iterations.

train_model_Mini5(10)

#CREATE A PLOT ERROR SURDACE OBJECT
get_surface=plot_error_surfaces(15,13,X,Y,30,go=False)

#create dataloader object and data object
dataset=Data()
trainloader=DataLoader(dataset=dataset,batch_size = 5)
#define train_model_mins function
w=torch.tensor(-15.0,requires_grad = True)
b=torch.tensor(-10.0,requires_grad = True)
LOSS_MINI10=[]
lr=0.1
def train_model_Mini10(epochs):
    for epoch in range(epochs):
        Yhat = forward ( X )
        get_surface.set_para_loss ( w.data.tolist ( ) , b.data.tolist ( ) , criterion ( Yhat , Y ).tolist ( ) )
        get_surface.plot_ps ( )
        LOSS_MINI10.append ( criterion ( forward ( X ) , Y ).tolist ( ) )
        for x , y in trainloader :
            yhat = forward ( x )
            loss = criterion ( yhat , y )
            get_surface.set_para_loss ( w.data.tolist ( ) , b.data.tolist ( ) , loss.tolist ( ) )
            loss.backward ( )
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data
            w.grad.data.zero_ ( )
            b.grad.data.zero_ ( )

#run the train_model_minis with 10 iteration
# train_model_Mini10(10)

#PLOT OUT THE LOSS FOR EACH METHOD
LOSS_BGD=[loss.detach().numpy() for loss in LOSS_BGD]
plt.plot(LOSS_BGD,label="Batch gradient descent")
plt.plot(LOSS_SGD,label="Stochastic gradient descent")
plt.plot(LOSS_MINI5,label = "Mini-Batch Gradient Descent, Batch size: 5")
plt.plot(LOSS_MINI10,label = "Mini-Batch Gradient Descent, Batch size: 10")
plt.legend()