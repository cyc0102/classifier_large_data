'''
This is a demo python program to demo python and the important module for Image Recogniton using CNN


Author: Bryan Chen

'''


import numpy as np

i=10
Img_path = 'data/test2/'+str(i)+'.jpg'
print (Img_path)

from keras.preprocessing.image import  img_to_array, load_img
img = load_img(Img_path,target_size=(150,150)) # PIL image
x = img_to_array(img)    # this is a Numpy array with shape ( Y, X , 3)
print(x.shape)
print(x.dtype)
x = x.astype('float32') / 255.0
print(x.dtype)

import matplotlib.pyplot as plt 
fig = plt.gcf()
fig.set_size_inches(10, 5)
plt.imshow(x)             # RGB type 0~255 int or 0~1 float
plt.show()

x1=x.reshape(1,150,150,3)
print('Reshape = ', x1.shape)


from keras.models import load_model

# 載入模型
model = load_model('model_3.h5') # trained by large data
prediction=model.predict(x1)
print('The prediction value is', prediction[0])
if (prediction[0] > 0.5):  
    print('It is a dog image!')
else:
    print('It is a cat image!')  

import numpy as np
x_4d=np.zeros((10,150,150,3),dtype=float)  # create 4 dimention ndarray with elements of zero
print('Dimmentions =',x_4d.ndim) 
# x_4d[0]=x
# print(x_4d)

index_str= input('Input the first index(1~12490) of test image :') #input is a str
batch_size = 10
for i in range (batch_size):
    index= int(index_str) + i # transfer index type to int and plus i (0~9)
    Img_path = 'data/test2/'+ str(index)  +'.jpg'
    img = load_img(Img_path,target_size=(150,150)) 
    x = img_to_array(img)    # this is a Numpy array with shape ( Y, X , 3)
    x = x.astype('float32') / 255.0
    x_4d[i]=x 
prediction=model.predict(x_4d)
print('The prediction value is', prediction[:10])
