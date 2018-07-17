from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# dimensions of our images.
img_width, img_height = 150, 150
test_data_dir = 'data/test1'
nb_validation_samples = 200
batch_size = 1

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_data = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')


# 載入模型
#model = load_model('model_3.h5') # trained by large data

# model.summary()
#prediction=model.predict_generator(test_data)
#print('First 20 images are', prediction[:20])
#print ('input shape =', test_data.image_shape)
#print ('test data =', test_data[0])

model = load_model('scrip3_model.h5') # trained by little data
scores=model.evaluate_generator(test_data)
print('ACC (little data model) =', scores[1])
model = load_model('model_3.h5') # trained by large data
scores=model.evaluate_generator(test_data)
print('ACC (large data model) =', scores[1])
