#PREBUILT DATASETS AND TRANSFORMS
import  torch
import matplotlib.pylab as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dsets
torch.manual_seed(0)
#show data by diagram
def show_data(data_sample,shape=(28,28)):
    plt.imshow(data_sample[0].numpy().reshape(shape),cmap = 'gray')
    plt.title('y='+str(data_sample[1]))
    plt.show()
dataset=dsets.MNIST(
    root='./data',
    download=False,
    transform=transforms.ToTensor()
)

#examining the daataset
# print("The type of element:",type(dataset[0]))
# print("The length of the tuple:", len(dataset[0]))
# print("The shape of the first element in tuple:",dataset[0][0].shape)
# print("The type of the first element in the tuple:",type(dataset[0][0]))
# print("The second element in the tuple:",dataset[0][1])
# print("The type of the second element in the tuple:",type(dataset[0][1]))
# print("As the result, the structure of the first element in the dataset is:,tensor([1,28,28]),tensor(7)")

#PLOTTING THE FIRST ELEMENT IN THE DATASET
# show_data(dataset[0])
#PLOT THE SECOND ELEMENT IN THE DATASET
# show_data(dataset[1])

#TORCH VISION TRANSFORMS
#combine two transforms:crop and convert to tensor . apply the compose to MNIST dataset
croptensor_data_transform=transforms.Compose([transforms.CenterCrop(20),transforms.ToTensor()])
dataset=dsets.MNIST(root = './data',download = True,transform = croptensor_data_transform)
# print("The shape of the first element in the first tuple:",dataset[0][0].shape)
#plot the first element in the dataset
# show_data(dataset[0],shape = (20,20))
#plot the second element in the dataset
# show_data(dataset[1],shape = (20,20))


#construct the compose.apply it on MNIST dataset. plot the image out
fliptensor_data_transform=transforms.Compose([transforms.RandomHorizontalFlip(p=1),transforms.ToTensor()])
dataset=dsets.MNIST(root = './data',download = True,transform = fliptensor_data_transform)
# show_data(dataset[1])


#combining vertical flip,horizontal flip and convert to tensor as a compose. apply the compose on image. Then plot the image
my_data_transform=transforms.Compose([transforms.RandomVerticalFlip(p=1),transforms.RandomHorizontalFlip(p=1),transforms.ToTensor()])
dataset=dsets.MNIST(root = './data',train = False,download = True,transform = my_data_transform)
show_data(dataset[1])
