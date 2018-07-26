'''
This is a demo python program to demo python and the important module for Image Recogniton using CNN


Author: Bryan Chen

'''
from keras.models import load_model     # from module keras.models import load_model
import numpy as np
from keras.preprocessing.image import  img_to_array, load_img
import matplotlib.pyplot as plt 

# 載入模型
model = load_model('model_3.h5')        # trained by large data
model.summary()                         #print model summary

label_dict={0:'cat', 1:'dog'}



batch_size = 12
x_4d=np.zeros((batch_size,150,150,3),dtype=float)  # create 4 dimention ndarray with elements of zero
print('x_4d.shape=',x_4d.shape)
print('x_4d.ndim =',x_4d.ndim) 

def plot_12_images(images,prediction):
    fig = plt.gcf()
    fig.set_size_inches(12,12)
    for i in range(batch_size):
        ax=plt.subplot(3,4,i+1)
        ax.imshow(images[i])
        index=int(index_str)+i
        title='Index=' + str(index) +' ' +'Result=' + label_dict[int(prediction[i])]
        ax.set_title(title,fontsize = 10)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()



while 1:
    index_str= input('Input Images  (index:1~189, batch_size=12, out of range for exit):') #input is a str
    index=int(index_str)
    if (index<1 or index>189):
        break

    for i in range (batch_size):
        
        Img_path = 'data/test3/'+ str(index)  +'.jpg'
        img = load_img(Img_path,target_size=(150,150)) 
        x = img_to_array(img)    # this is a Numpy array with shape ( Y, X , 3)
        x = x.astype('float32') / 255.0
        x_4d[i]=x 
        index= index +1 
    
    prediction=model.predict(x_4d)
    prediction=np.rint(prediction)
    plot_12_images(x_4d,prediction)


