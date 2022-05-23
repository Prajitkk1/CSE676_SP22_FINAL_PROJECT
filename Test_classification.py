# -*- coding: utf-8 -*-


import pandas as pd

from gan_utils import MyModel
import tensorflow as tf

data = pd.read_csv("D:/courses/deep learning/final-project_moreiterations/dataset/Test.csv") 
data["ClassId"] = data["ClassId"].astype(str)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

test_generator = train_datagen.flow_from_dataframe(
    data, 
    'dataset/', 
    x_col='Path',
    y_col='ClassId',
    target_size=(124, 124),
    class_mode='categorical',
    batch_size=8
)


# model = MyModel(43)

##################################Accuracy of default model############################
loaded_model = tf.keras.models.load_model('best_model_classification_default/best_model_resnet_1')
pred = loaded_model.evaluate_generator(test_generator)

print("Accuracy = ", pred[1])




##################################Accuracy of GAN model############################
loaded_model = tf.keras.models.load_model('best_model_classification_withGAN/best_model_resnet_1')
pred = loaded_model.evaluate_generator(test_generator)

print("Accuracy = ", pred[1])



