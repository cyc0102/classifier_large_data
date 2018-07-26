'''
This is a demo python program to demo python and the important module for Image Recogniton using CNN


Author: Bryan Chen

'''
from keras.models import load_model


# 載入模型
model = load_model('model_3.h5')                    # trained by large data
model.summary()                                     # print model summary

#GUI for image input
from easygui import fileopenbox, msgbox
msgbox('Select an image file in data directory')
Img_path=fileopenbox()
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

# 輸出格式轉換
import numpy as np
prediction=np.rint(prediction)                     # transfer ndarray item to nearest int
print('np.rint(prediction)=',prediction)           
print('prediction.shape=',prediction.shape)
print('prediction.dtype=',prediction.dtype)
prediction=prediction.astype(int)
print('prediction=',prediction)
print('prediction.dtype=',prediction.dtype)
print('prediction[0]=',prediction[0])
i=prediction[0]
print('i=',i)
print('i.shape=',i.shape)
print('i.dtype=',i.dtype)
i=int(i)                                           # transfer ndarray to python int
print('int(i)=',i,'type(i)=',type(i))
label_dict={0:'Cat', 1:'Dog'}                      # dictionary {key: value,...}    
in_result=label_dict[i]                            # value = dict[key]
print('in_result=',in_result)

# 以圖形輸出結果
import matplotlib.pyplot as plt 
def plot_a_image(image,title,result):
    fig = plt.gcf()
    fig.set_size_inches(5, 6)
    plt.imshow(x)                                   # RGB type 0~255 int or 0~1 float
    plt.title(title,fontsize=10)
    plt.text(20,170,'The prediction result --> '+ result,fontsize=15)
    plt.show()

in_title='input Image:' + Img_path  
plot_a_image(x,in_title,in_result)


