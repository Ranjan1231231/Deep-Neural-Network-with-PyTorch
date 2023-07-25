import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
a=torch.tensor([0,1,2,3,4,5])#CREATING A TENSOR
# print(a.dtype)#DATA TYPE OF ELEMENTS IN A TENSOR
# print(a.type())#DATA TYPE OF A
b=torch.FloatTensor([0,0.1,0.2,0.3])#CREATING SPECIFIC DATA TYPE TENSOR
# print(b.type())

#CHANGING THE TYPE OF THE TENSOR
a=torch.tensor([0,1,2,3,4])
a=a.type(torch.FloatTensor)
# print(a.type())
# print(torch.FloatTensor)


a=torch.Tensor([0,1,2,3,4])
# print(a.size())
# print(a.ndimension())
a_col=a.view(5,1)#5 for rows, 1 for cols
# print(a_col)


#converting numpy array to torch tensor
numpy_array=np.array([0.0,1.0,2.0,3.0,4.0])
torch_tensor=torch.from_numpy(numpy_array)
back_to_numpy: object=torch_tensor.numpy()


#CONVERTING PANDA SERIES TO TENSOR
pandas_Series=pd.Series([0.1,2,0.3,10.1])
pandas_to_torch=torch.from_numpy(pandas_Series.values)


#INDEXING AND SLICING
c=torch.tensor([20,1,2,3,4])
c[0]=100
c[4]=0
# print(c)

d=c[1:4]
# print(d)
c[3:5]=torch.tensor([300.0,400.0])
# print(c)

#BASIC OPERATIONS
#vector addition and subtraction
u=torch.tensor([1.0,0.0])
v=torch.tensor([0.0,1.0])
z=u+v#tensor should be of same type
p=u-v#tensor should be of same type
# print(z,p)

#vector multiplication with scaler
y=torch.tensor([1,2])
z=2*y
# print(z)

#product of two tensors

u=torch.tensor([1,2])
v=torch.tensor([3,2])
z=u*v
# print(z)

#dot product#it provides  how similar two tensors are
u=torch.tensor([1,2])
v=torch.tensor([3,1])
result=torch.dot(u,v)#(1X3)+(2X1)
# print(result)


#ADDING CONSTANT TO A TENSOR(BRODCASTING)
u=torch.tensor([1,2,3,-1])
z=u+1
# print(z)


#UNIVERSAL FUNCTION
#MEAN
# a=torch.tensor([1,1,1,1])
# mean_a=a.mean()
# print(mean_a)

#MAX
b=torch.tensor([1,-2,3,4,5])
max_b=b.max()
# print(max_b)

#MAPPING TENSORS TO NEW TORCH TENSORS
x=torch.tensor([0,np.pi/2,np.pi])
y=torch.sin(x)
# print(y)

#LINESPACE
x=torch.linspace(0,2*np.pi,100)
y=torch.sin(x)
# plt.plot(x.numpy(),y.numpy())
# plt.show()


#this function is for plotting diagrams
# Plot vecotrs, please keep the parameters in the same length
# @param: Vectors = [{"vector": vector variable, "name": name of vector, "color": color of the vector on diagram}]


def plotVec(vectors):
    ax=plt.axes()
    #for loo to draw the vectors
    for vec in vectors:
        ax.arrow(0,0,*vec["vector"],head_width=0.05,color=vec["color"],head_length=0.1)
        plt.text(*(vec["vector"]+0.1),vec["name"])
    plt.ylim(-2,2)
    plt.xlim(-2,2)

#TYPES AND SHAPES
#CONVERT A INTEGER LIST  WITH LENGTH 5 TO A TENSOR
ints_to_tensor=torch.tensor([0,1,2,3,4])
# print("The dtype of the tensor object after converting it to tensor",ints_to_tensor.dtype)
# print("The type of tensor object after converting it to tensor",ints_to_tensor.type())

#as a result a the integer has been converted to a long tensor
# print(type(ints_to_tensor))


#CONVERT A FLOAT LIST WITH LENGTH 5 TO A TENSOR
floats_to_tensor=torch.tensor([0.0,1.0,2.0,3.0,4.0])
# print("the dtype of tensor objec after converting it to a tensor:",floats_to_tensor.dtype)
# print("the type of tensor object after converting it to tensor:",floats_to_tensor.type())


list_floats=[0.0,1.0,2.0,3.0,4.0]
floats_int_tensor=torch.tensor(list_floats,dtype = torch.int64)
# print("the dtype of tensor object is :",floats_int_tensor.dtype)
# print("the type of tensor object is :",floats_int_tensor.type())


#CONVERT A INTEGER LIST WITH LENGTH 5 TO FLOAT TENSOR
new_float_tensor=torch.FloatTensor([0,1,2,3,4])
print(new_float_tensor.type())
# print("the type of the new_float_tensor:",new_float_tensor.type())
new_float_tensor=torch.FloatTensor([0,1,2,3,4])
print(new_float_tensor.type())

#ANOTHER METHOD TO CONVERT THE INTEGER LIST TO FLOAT TENSOR
old_int_tensor=torch.tensor([0,1,2,3,4])
new_float_tensor=old_int_tensor.type(torch.FloatTensor)
print("the type of the new_float_Tensor:",new_float_tensor.type())


#INTRODUCE THE TENSOR_OBJ.SIZE()&TENSOR_NDIMENSION.SIZE() METHODS
print("the size of the new_float_tensor:",new_float_tensor.size())
print("the dimension of the new_float_tensor:",new_float_tensor.ndimension())

#INTRODUCE THE TENSOR_OBJ.VIEW(ROW,COLUMN)METHOD
twoD_float_tensor=new_float_tensor.view(5,1)
print("Orignal Size:",new_float_tensor)
print('size after view method',twoD_float_tensor)

#INTRODUCE THE USE OF -1 IN TENSOR_OBJ.VIEW(ROW,COLUMN)METHOD
twoD_float_tensor=new_float_tensor.view(-1,1)#fromback side
print("Oiginal size:",new_float_tensor)
print("Size after view method",twoD_float_tensor)

#convert  a numpy array to  tensor
numpy_array=np.array([0.0,1.0,2.0,3.0,4.0])
new_tensor=torch.from_numpy(numpy_array)
print("the dtype of new tensor:" ,new_tensor.dtype)
print("the type of new tensor:",new_tensor.type())

#CONVERT A TENSOR TO A NUMPY ARRAY
back_to_numpy = new_tensor.numpy()
print("THE NUMPY ARRAY FROM TENSOR: ",back_to_numpy)
print("the dtype of numpy array: ",back_to_numpy.dtype)

#SET ALL ELEMENTS IN NUMPY ARRAY TO ZERO
numpy_array[:]=0
print("the new tensor points to numpy_array: ",new_tensor)
print("and  back to nump array points to the tensor:",back_to_numpy)


#CONVERT A PANDA SERIES TO A TENSOR
pandas_series=pd.Series([0.1,2,0.3,10.1])
new_tensor=torch.from_numpy(pandas_series.values)
print("the new tensor from numpy array:",new_tensor)
print("the datatype of new tensor:",new_tensor.dtype)
print("the type of new tensor:",new_tensor.type())


this_tensor=torch.tensor([0,1,2,3])
print("the first item is given by",this_tensor[0].item(),"the first rensor is given by ",this_tensor[0])
print("the second item is given by", this_tensor[1].item(),"the second tensor is given by ", this_tensor[1])
print("the third item is given by",this_tensor[2].item(),"the third tensor value is given by",this_tensor[2])


torch_to_list=this_tensor.tolist()
print('tensor:',this_tensor,"\nlist:",torch_to_list)



my_tensor=torch.tensor([1,2,3,4,5])
print(my_tensor.view(5,1))#converting mytensor to 5 columns and 1 row each columns

#indexing and slicing
index_tensor=torch.tensor([0,1,2,3,4])
print("the value on the index 0:",index_tensor[0])
print("the value on the index 1:",index_tensor[1])
print("the value on the index 2:",index_tensor[2])
print("the value on the index 3:",index_tensor[3])
print("the value on the index 4:",index_tensor[4])

#A tensor for showing how to change value according to the index
tensor_sample=torch.tensor([20,1,2,3,4])
#change the value on the index 0 to 100
print('inital value on index 0:',tensor_sample[0])
tensor_sample[0]=100
print("modified tensor:",tensor_sample)

#changethe value on the index 4 to 0
print("initial value on index 4:",tensor_sample[4])
tensor_sample[4]=0
print("Modified tensor:",tensor_sample)


#slice tensor_Sample
subset_tensor_sample=tensor_sample[1:4]
print("orignal tensor sample:",tensor_sample)
print("the subset of tensor sample:",subset_tensor_sample)

#change the value on index 3 and index 4
print("inital value on index 3 and index 4:",tensor_sample[3:5])
tensor_sample[3:5]=torch.tensor([300.0,400.0])
print("Modified tensor:",tensor_sample)

#using variable to assign the value to the seleccted values
print("the inital tensor_sample",tensor_sample)
selected_indexes=[1,3]
tensor_sample[selected_indexes]=100000
print("modeified tensor with one value:",tensor_sample)


#practice
practice_tensor=torch.tensor([2,7,3,4,6,2,3,1,2])
selected_indexes=[3,4,7]
practice_tensor[selected_indexes]=0
print("modified tensor with one value:",practice_tensor)


#tensor functions
#mean and standard deviation
#sample tensor for mathematic calucation methods on tensor
math_tensor=torch.tensor([1.0,-1.0,-1])
print("tensor example:",math_tensor)
#calculate the mean for math tensor
mean=math_tensor.mean()
print("THE MEAN OF MATH_TENSOR:",mean)

#calculate the standard deviation for math_tensor
standard_deviation=math_tensor.std()
print("the standard deviation of math_tensor:",standard_deviation)

#max and min
#sample for  intoducing max and min methods
max_min_tensor=torch.tensor([1,1,3,5,5])
print("tensor example:",max_min_tensor)
#method for finding the maximum value in the tensor
max_val=max_min_tensor.max()
print("maximumnumber in the tensor:",max_val)
#method for finding the minimum value in the tensor
min_val=max_min_tensor.min()
print("minimum value of the tensor:",min_val)

#METHOD FOR CALCULATING THE SIN RESULTOF EACH ELEMENT IN THE TENSOR
pi_tensor=torch.tensor([0,np.pi/2,np.pi])
sin=torch.sin(pi_tensor)
print("the sin of pi_tensor:",sin)
temp_tensor=torch.tensor([0,1,2,3,4,5,6,7,8,9,])
sin=torch.sin(temp_tensor)
print(sin)


#CREATING THE TENSOR BY TORCH.LINSPACE()
len_5_tensor=torch.linspace(-2,3,steps = 5)
print("first try on linspace",len_5_tensor)
len_9_tensor=torch.linspace(-2,3,steps=9)
print("second try on linspace:",len_9_tensor)


#constructing a tensor within 0 to 360 degree
pi_tensor=torch.linspace(0,2*np.pi,100)
sin_result=torch.sin(pi_tensor)
# plt.plot(pi_tensor.numpy(),sin_result.numpy())
# plt.show()

#practice
pi_tensor = torch.linspace(0, np.pi/2, 100)
print("Max Number: ", pi_tensor.max())
print("Min Number", pi_tensor.min())
sin_result1 = torch.sin(pi_tensor)
# plt.plot(pi_tensor.numpy(), sin_result1.numpy())
#
# plt.show()



#tensor operations
#tensor addition
u=torch.tensor([1,0])
v=torch.tensor([0,1])
w=u+v
print("the result tensor:",w)
#plot u,v,w
plotVec([{"vector":u.numpy(),"name":'u','color':'r'},
         {"vector" : v.numpy ( ) , "name" : 'v' , 'color' : 'b'},
         {"vector" : w.numpy ( ) , "name" : 'w' , 'color' : 'g'}
         ])

#TENSOR SUBTRACTION IMPLIMENTATION
u=torch.tensor([1,0])
v=torch.tensor([0,1])
s=u-v
print("the result tensor:" , s)
#adding a scalar value to tensor
u=torch.tensor([1,2,3,-1])
v=u+1
print("SCALAR VALUE ADDITION RESULT:",v)

# tensor multiplication
#tensor *scalar
u=torch.tensor([1,2])
v=2*u
print("the result is" ,v)
#tensor * tensor
u=torch.tensor([1,2])
v=torch.tensor([3,2])
w=u*v
print("the result of u*v",w)

#dot product
u=torch.tensor([1,2])
v=torch.tensor([3,2])
print("dot product of u,v:",torch.dot(u,v))#1x3+2x2
