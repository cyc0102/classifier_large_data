from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
'''
This is a demo python program to demo python and the important module for Image Recogniton using CNN


Author: Bryan Chen

'''
import numpy as np

i=1
Img_path = 'data/test2/'+str(i)+'.jpg'
print (Img_path)

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
img = load_img(Img_path,target_size=(150,150))
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

x=x.reshape(1,150,150,3)
print('Reshape = ', x.shape)


from keras.models import load_model

# 載入模型
model = load_model('model_3.h5') # trained by large data
prediction=model.predict(x)
print('The prediction value is', prediction[0])
if (prediction[0] > 0.5):  
    print('It is a dog image!')
else:
    print('It is a cat image!')  

import numpy as np
x_4d=np.zeros((10,150,150,3),dtype=float)  # create 4 dimention ndarray with elements of zero
print('Dimmentions =',x_4d.ndim) 
x_4d[0]=x
print(x_4d)
