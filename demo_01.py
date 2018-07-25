'''
This is a demo python program to demo python and the important module for Image Recogniton using CNN


Author: Bryan Chen

'''
from keras.models import load_model

# 載入模型
model = load_model('model_3.h5')                    # trained by large data
model.summary()                                     # print model summary

# 輸入檔案
i_str=input('Input Image (index:1~200):')           # Input type is string 
i=int(i_str)                                        # change type to int
if (i<1 or i>200):
    exit()
Img_path = 'data/test3/'+str(i)+'.jpg'
print('Image file path:',Img_path)
from keras.preprocessing.image import  img_to_array, load_img
img_org = load_img(Img_path)                        # PIL image
img_org.show()                                      # show the original PIL image
img = load_img(Img_path,target_size=(150,150))      # PIL image, targetz_size for above model

# 將image file轉換為CNN 輸入格式
x = img_to_array(img)                               # a Numpy array with shape ( Y, X , 3)
print('x.shpae=',x.shape)                           # (150,150,3)
print('x.dtype=',x.dtype)
x = x / 255.0                                       # normalize
x1=x.reshape(1,150,150,3)
print('Image after Reshape = ', x1.shape)

# 執行預測
prediction=model.predict(x1)
print('The prediction value is', prediction[0])
if (prediction[0] > 0.5):  
    print('It is a dog image!')
else:
    print('It is a cat image!')  


import numpy as np
prediction=np.rint(prediction)
print('np.rint(prediction)=',prediction)
print(prediction.shape)
print(prediction.dtype)
prediction=prediction.astype(int)
print(prediction)
print(prediction.dtype)
print('prediction[0]=',prediction[0])

label_dict={0:'Cat', 1:'Dog'}

import matplotlib.pyplot as plt 

def plot_a_image(image,title,result):
    fig = plt.gcf()
    fig.set_size_inches(5, 6)
    plt.imshow(x)             # RGB type 0~255 int or 0~1 float
    plt.title(title,fontsize=12)
    plt.text(20,170,'The prediction result --> '+ result,fontsize=15)
    plt.show()


in_title='input Image:' + Img_path  

i=prediction[0]
i=int(i)
in_result=label_dict[i]
print(in_result)
plot_a_image(x,in_title,in_result)


'''
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
prediction=np.rint(prediction)
print('The prediction value is', prediction[:10])
'''