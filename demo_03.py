'''
This is a demo python program to demo python and the important module for Image Recogniton using CNN


Author: Bryan Chen

'''
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import  img_to_array, load_img
import matplotlib.pyplot as plt 

# 載入模型
model = load_model('model_3.h5') # trained by large data
model.summary()  #print model summary

label_dict={0:'cat', 1:'dog'}

def plot_a_image(image,title,result):
        fig = plt.gcf()
        fig.set_size_inches(5, 6)       # set size
        plt.imshow(x)                   # RGB type 0~255 int or 0~1 float
        plt.title(title,fontsize=12)
        plt.text(20,170,'The prediction result --> '+ result,fontsize=15)
        plt.show()

while 1:

    i_str=input('Input Image (index:1~12500, out of range for exit):')
    i=int(i_str)
    if (i<1 or i>12500):
        break
    Img_path = 'data/test2/'+str(i)+'.jpg'
    print('Imge file path:',Img_path)
 
    img_org = load_img(Img_path)                    # PIL image
    print('The original image!')
    img_org.show()                                  # Show the original image
    img = load_img(Img_path,target_size=(150,150))  # PIL image, target_size=(150,150)
    # print('The target_size image!')
    # img.show()
    x = img_to_array(img)                           # this is a Numpy array with shape ( Y, X , 3)
    print('x.shape=',x.shape)
    print('x.dtype',x.dtype)
    x = x.astype('float32') / 255.0

    x1=x.reshape(1,150,150,3)
    print('x.reshape = ', x1.shape)


    prediction=model.predict(x1)
    print('The prediction value is', prediction[0])
    if (prediction[0] > 0.5):  
        print('It is a dog image!')
    else:
        print('It is a cat image!')  

    prediction=np.rint(prediction)
    print('np.rint(prediction)=',prediction)
    print('prediction.shape=',prediction.shape)
    print('predicton.dtype=',prediction.dtype)
    prediction=prediction.astype(int)               #change Numpy array type
    print('predicton after chagne type to int',prediction)
    print('prediction.shape=',prediction.shape)
    print('prediction.dtype=',prediction.dtype)
    print('prediction[0]=',prediction[0])
   

    in_title='input Image:' + Img_path  

    i=prediction[0]
    print('i=',i)
    i=int(i)                                        #change type to pyton int
    print('int(i)=',i)
    in_result=label_dict[i]
    print('in_result=',in_result)
    plot_a_image(x,in_title,in_result)



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


batch_size = 12
while 1:
    index_str= input('Input Images  (index:1~12489, batch_size=12, out of range for exit):') #input is a str
    index=int(index_str)
    if (index<1 or index>12489):
        break

    for i in range (batch_size):
        
        Img_path = 'data/test2/'+ str(index)  +'.jpg'
        img = load_img(Img_path,target_size=(150,150)) 
        x = img_to_array(img)    # this is a Numpy array with shape ( Y, X , 3)
        x = x.astype('float32') / 255.0
        x_4d[i]=x 
        index= index +1 
    
    prediction=model.predict(x_4d)
    prediction=np.rint(prediction)
    print('The prediction value is', prediction[:batch_size])
    plot_12_images(x_4d,prediction)


