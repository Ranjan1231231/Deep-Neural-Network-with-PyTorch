import torch
from  torch.utils.data import Dataset
from torchvision import transforms
torch.manual_seed(1)
#SIMPLE DATASET
class toy_set(Dataset):
    #constructor with default values
    def __init__(self,length=100,transform=None):
        self.len=length
        self.x=2*torch.ones(length,2)
        self.y=torch.ones(length,1)
        self.transform=transform
    #getter
    def __getitem__( self,index ):
        sample=self.x[index],self.y[index]
        if self.transform:
            sample=self.transform(sample)
        return sample
    #get length
    def __len__(self):
        return self.len


#create dataset object find out the values on index 1 , findout the length of dataset object
our_dataset=toy_set()
# print("Our toy_set object:",our_dataset[0])
# print("Value on index 0 of our toy_set object:",our_dataset[0])
# print("our toy_Set length:",len(our_dataset))

#usig loop to print out first 3 elements in dataset
# for i in range(3):
#     x,y=our_dataset[i]
#     print("index",i, '; x:', x, '; y:', y)
#
# for x, y in our_dataset:
#     print("x:",x,"y:",y)
#
#creating a object with length 50
my_dataset=toy_set(length = 50)
# print('my toy set length:',len(my_dataset))

#TRANSFORMS

#creating a transform class add_mult
class add_mult(object):
    #constructor
    def __init__(self,addx=1,muly=2):
        self.addx=addx
        self.muly=muly
    #executor
    def __call__(self, sample):
        x=sample[0]
        y=sample[1]
        x=x+self.addx
        y=y+self.muly
        sample = x,y
        return sample

#creating a add_mult transformobject , adn a toy_Set object
a_m=add_mult()
data_set=toy_set()
#using loop to print out first 10 elements in dataset
# for i in range(10):
#     x,y=data_set[i]
#     print("Index:",i,'Orignal x:',x,'orignal y:',y)
#     x_,y_=a_m(data_set[i])
#     print('Index:',i,'Trandformed x_:',x_,'Transformed y_:',y_)


#creating a new dataset object with add_mjult object as transform
cust_data_set=toy_set(transform = a_m)

#using loop to print out first 1- elements in dataset
# for i in range(10):
#     x,y=data_set[i]
#     print("Index:",i,"Orignal x:",x,"orignal y:",y)
#     x_,y_=cust_data_set[i]
#     print('Index:',i,'Transformed x_:',x_,'Transformed y_:',y_)
#


##compose

#create a new transform class that multiplies each of the elements by 100:
# Create tranform class mult

class mult ( object ) :

    # Constructor
    def __init__ ( self , mult = 100 ) :
        self.mult = mult

    # Executor
    def __call__ ( self , sample ) :
        x = sample [ 0 ]
        y = sample [ 1 ]
        x = x * self.mult
        y = y * self.mult
        sample = x , y
        return sample
# Combine the add_mult() and mult()

data_transform = transforms.Compose([add_mult(), mult()])
print("The combination of transforms (Compose): ", data_transform)
data_transform(data_set[0])

x,y=data_set[0]
x_,y_=data_transform(data_set[0])
print( 'Original x: ', x, 'Original y: ', y)

print( 'Transformed x_:', x_, 'Transformed y_:', y_)

# Create a new toy_set object with compose object as transform

compose_data_set = toy_set(transform = data_transform)
# Use loop to print out first 3 elements in dataset

for i in range(3):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = cust_data_set[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)
    x_co, y_co = compose_data_set[i]
    print('Index: ', i, 'Compose Transformed x_co: ', x_co ,'Compose Transformed y_co: ',y_co)
