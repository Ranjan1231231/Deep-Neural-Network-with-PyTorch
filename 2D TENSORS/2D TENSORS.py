import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#CONVERT 2D LIST TO TENSOR
two_D_list=[[11,12,13],[21,22,23],[31,32,33]]
two_D_tensor=torch.tensor(two_D_list)
# print("The new 2D tensor:",two_D_tensor)
# print("The dimension of 2D tensor:",two_D_tensor.ndimension())
# print("The shape of 2D tensor:",two_D_tensor.shape)
# print("The size of 2D tensor:",two_D_tensor.size())
# print("The number of elements in 2D tensor:", two_D_tensor.numel())

#Convert tensor to numpy array, convert numpy array to tensor
twod_numpy=two_D_tensor.numpy()
# print("The numpy array after converting: ", twod_numpy)
# print("Type after converting: ", twod_numpy.dtype)
new_twod_tensor=torch.from_numpy(twod_numpy)
# print("The tensor after converting:", new_twod_tensor)
# print("Type after converting: ", new_twod_tensor.dtype)


#CONVERTING PANDA DATAFRAME TO TENSOR
df=pd.DataFrame({'a':[11,21,31],'b':[12,22,32]})
# print("Pandas Dataframe:",df.values)
# print("df type:",df.values.dtype)
new_tensor=torch.from_numpy(df.values)
# print("Tensor after converting:",new_tensor)
# print("Type after converting:",new_tensor.dtype)
df = pd.DataFrame({'A':[11, 33, 22],'B':[3, 3, 2]})
converted_tensor = torch.tensor(df.values)
# print ("Tensor: ", converted_tensor)



#INDEXING AND SLICING

#indexing
tensor_example=torch.tensor([[11,12,13],[21,22,23],[31,32,33]])
# print("THE VALUE OF 2ND ROW AND 3RD COLUMN",tensor_example[1,2])#starts from 0 so we did n-1
# print("THE VALUE 2ND ROW AND 3RD COLUMN IN ANOTHER WAY",tensor_example[1][2])

#slicing
# print("THE VALUE OF  1ST ROW AND FIRST TWO COLUMNS",tensor_example[0,0:2])
# print("THE VALUE OF 1ST ROW AND FIRST TWO COLUMNS IN ANOTHER WAY",tensor_example[0][0:2])

#WE CANNOT APPLY TENSOR[0:1] BECAUSE IT WILL PRODUCE A 2D MATRIX AGAIN
sliced_tensor_example=tensor_example[1:3]
# print("1. Slicing step on tensor_example: ")
# print("Result after tensor_example[1:3]: ", sliced_tensor_example)
# print("Dimension after tensor_example[1:3]: ", sliced_tensor_example.ndimension())
# print("2. Pick an index on sliced_tensor_example: ")
# print("Result after sliced_tensor_example[1]: ", sliced_tensor_example[1])
# print("Dimension after sliced_tensor_example[1]: ", sliced_tensor_example[1].ndimension())
# print("3. Combine these step together:")
# print("Result: ", tensor_example[1:3][1])
# print("Dimension: ", tensor_example[1:3][1].ndimension())


## Use tensor_obj[begin_row_number: end_row_number, begin_column_number: end_column number]
# print("THE VALUE ON 3RD COLUMN AND LAST TWO ROWS",tensor_example[1:3,2])#we take 3 because while slicing it doesnt display the last element so we have to take n+1

#CHANGING THE VALUE OF TENSOR ELEMENTS
tensor_example[1:3,1]=0
# print("THE NEW MODIFIED TENSOR IS:", tensor_example)



#TENSOR OPERATIONS
#tensor addition
X=torch.tensor([[1,0],[2,2]])
Y=torch.tensor([[2,1],[0,3]])
XandY=X+Y
# print("THE RESULT OF X+Y:",XandY)

#scalar multiplication
two_Y=2*Y
# print("THE RESULT OF 2*Y:",two_Y)

#tensor product(element wise multiplication/hadamard product)
X_times_Y=X*Y
# print("THE RESULT OF X*Y:",X_times_Y)


#tensor matrix multiplication
A = torch.tensor([[0, 1, 1], [1, 0, 1]])
B = torch.tensor([[1, 1], [1, 1], [-1, 1]])
A_times_B=torch.mm(A,B)# it takes the dot product of the two matrix(if one matrix is in MXN shape then the other matrix should be in NXM shape(eg.2X3 AND 3X2))
# print("THE VALUES OF A*B:",A_times_B)

#tensor matrix multiplication with same no of M AND N
a=torch.tensor([[8,2],[2,4]])
b=torch.tensor([[2,2],[3,5]])
# print(torch.mm(a,b))

#MULTIPLICATION OF TWO DIFFRENT SHAPE TENSOR
x=torch.tensor([[1,2],[4,3],[5,3]])
y=torch.tensor([[3,2,3,4],[2,4,5,4]])
print(torch.mm(x,y))


