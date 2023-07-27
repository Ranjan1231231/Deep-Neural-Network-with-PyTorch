#images link-"https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0110EN-SkillsNetwork/labs/Week1/data/img.tar.gz"
#xml file-"https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0110EN-SkillsNetwork/labs/Week1/data/index.csv "

import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
torch.manual_seed(0)
from matplotlib.pyplot import  imshow
import matplotlib.pylab as plt
from PIL import Image
import pandas as pd
import os
import torchvision.transforms as transforms
#Auxilarry functions
directory=""
csv_file='index.csv'
csv_path=os.path.join(directory,csv_file)

data_name=pd.read_csv(csv_path)
data_name.head()

def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + data_sample[1])
    plt.show()

#get the value on location row 0, column 1
# print('y:',data_name.iloc[0,1])

#get the value on location row 0, column 0
# print('y:',data_name.iloc[0,0])


#print out the file name and the class number of the element on row 1
# print("File name:",data_name.iloc[1,1])
# print('class or y:',data_name.iloc[1,0])

#print out the total number of rows in traning dataset
# print('the number of rows:',data_name.shape[0])


#load image

#combine the directory path with file name
# image_name=data_name.iloc[1,1]
# print(image_name)

#finding the image path
# image_path=os.path.join(directory,image_name)
# print(image_path)

#plot the second traning image
# image=Image.open(image_path)
# plt.imshow(image,cmap = 'gray',vmin = 0,vmax = 225)
# plt.title(data_name.iloc[1,0])
# plt.show()

#plot the 20th image
# image_name=data_name.iloc[19,1]
# image_path=os.path.join(directory,image_name)
# image=Image.open(image_path)
# plt.imshow(image,cmap = 'gray',vmin=0,vmax = 255)
# plt.title(data_name.iloc[19,0]
# plt.show()


#CREATE A DATASET CLASS
#creating ourown dataset object
class Dataset(Dataset):
    #constructor
    def __init__(self,csv_file,data_dir,transform=None):
        #image directory
        self.data_dir=data_dir
        #the transform is going to be used on image
        self.transform=transform
        data_dircsv_file=os.path.join(self.data_dir,csv_file)
        #Load the CSV file contains image info
        self.data_name=pd.read_csv(data_dircsv_file)
        #number of images in dataset
        self.len=self.data_name.shape[0]
    #get the length
    def __len__(self):
        return self.len
    #getter
    def __getitem__(self, idx):
        #image filepath
        img_name=os.path.join(self.data_dir,self.data_name.iloc[idx,1])
        #open image file
        image=Image.open(img_name)
        #the class label for the image
        y=self.data_name.iloc[idx,0]
        #if there is any transform method apply it nto the image
        if self.transform:
            image=self.transform(image)
        return image,y


#creating  the dataset objects
dataset=Dataset(csv_file=csv_file,data_dir = directory)
# image=dataset[0][0]
# y=dataset[0][1]
# plt.imshow(image,cmap = 'gray',vmin=0,vmax=255)
# plt.title(y)
# plt.show()
# print(y)
#
# image=dataset[9][0]
# y=dataset[9][1]
# plt.imshow(image,cmap = 'gray',vmin = 0,vmax=255)
# plt.title(y)
# plt.show()


#torchvision transforms

#combine two transforms : crop and convert to tensor, apply the compose to MNIST dataset
croptensor_Data_transforms=transforms.Compose([transforms.CenterCrop(20),transforms.ToTensor()])
dataset = Dataset(csv_file=csv_file , data_dir=directory,transform=croptensor_Data_transforms )
# print("The shape of the first element tensor:", dataset[0][0].shape)
#plot the first element in the dataset
show_data(dataset[0],shape=(20,20))
#plot the second element in the dataset
show_data(dataset[1],shape = (20,20))
#construct the compose , apply it on MNIST dataset, plot the image out
fliptensor_data_transform=transforms.Compose([transforms.RandomVerticalFlip(p=1),transforms.ToTensor()])
dataset=Dataset(csv_file=csv_file,data_dir =directory,transform = fliptensor_data_transform)
show_data(dataset[1])
