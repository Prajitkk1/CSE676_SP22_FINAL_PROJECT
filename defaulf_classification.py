# -*- coding: utf-8 -*-





import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
import cv2 as cv
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import preprocessing
import tensorflow as tf
from random import randint
import pandas as pd
import copy
from pathlib import Path
import tensorflow as tf


from tensorflow.python.keras.callbacks import ReduceLROnPlateau
#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
#sess = tf.Session(config=config) 
#keras.backend.set_session(sess)


from tensorflow.keras.models import load_model
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)



tf.config.run_functions_eagerly(True)
tf.config.experimental_run_functions_eagerly(True)
class DataGenerator(keras.utils.Sequence):
    
    def __init__(self,list_IDs, labels, batch_size=100, dim=(64,64,32),n_channels=3,n_classes=200,path = "/",shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channel = n_channels
        self.n_classes = n_classes
        self.path = path
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        print(int(np.floor(len(self.list_IDs)/self.batch_size)))
        return(int(np.floor(len(self.list_IDs)/self.batch_size)))
        
    def __getitem__(self, index):
        indices = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #print((indices))
        #list_IDs_temp = [self.list_IDs[k] for k in indices]
        list_IDs_temp = [k for k in indices]

        X, y = self.__data_generation(list_IDs_temp)
        return X, y
    
    def __data_generation(self,ids):
        #i = 0
        X = []
        Y = []
        for i in ids:
            img = cv.imread(self.list_IDs[i])
            img = img/255
            lab = self.labels[i]
            Y.append(lab)
            X.append(img)
            #i = i+1
        X,Y = np.array(X),keras.utils.to_categorical(Y, num_classes=self.n_classes)
        return X,Y            
#    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
            

            
class skip_block(layers.Layer):
    def __init__(self, filters,dims,**kwargs):
        super(skip_block,self).__init__(**kwargs)
        self.filters = filters
        self.dims = dims
        self.conv1 = layers.Conv2D(filters=self.filters,kernel_size=self.dims,padding='same')
        self.b1 = layers.BatchNormalization()
        self.act1 = layers.Activation('relu')
        self.conv2 = layers.Conv2D(filters=self.filters,kernel_size=self.dims,padding='same')
        self.conv3 = layers.Conv2D(filters=self.filters,kernel_size=(1,1),padding='same')
        self.b2 = layers.BatchNormalization()
        self.add1 = layers.Add()
        self.act2 = layers.Activation('relu')
        

    def call(self,inputs):
        x_copy = copy.copy(inputs)
        x = self.conv1(inputs)
        x = self.b1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.b2(x)
        x_copy = self.conv3(x_copy)
        x = self.add1([x,x_copy])
        x = self.act2(x)
        return x

class residual_block(layers.Layer):
    def __init__(self, filters,dims,**kwargs):
        super(residual_block,self).__init__()
        self.filters = filters
        self.dims = dims
        self.conv1 = layers.Conv2D(filters=self.filters,kernel_size=self.dims,padding='same')
        self.b1 = layers.BatchNormalization()
        self.act1 = layers.Activation('relu')
        self.conv2 = layers.Conv2D(filters=self.filters,kernel_size=self.dims,padding='same')
        self.b2 = layers.BatchNormalization()
        self.add1 = layers.Add()
        self.act2 = layers.Activation('relu')
        

    def call(self,inputs):
        x_copy = copy.copy(inputs)
        x = self.conv1(inputs)
        x = self.b1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.b2(x)
        x = self.add1([x,x_copy])
        x = self.act2(x)
        return x

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10),
    tf.keras.callbacks.ModelCheckpoint(filepath='best_model_classification_default/best_model_resnet_1',save_best_only=True,save_format='tf'),
    tf.keras.callbacks.TensorBoard(log_dir='.\logstf\classification_default'),
    tf.keras.callbacks.CSVLogger("model_history_classification_default.csv",append=True)
]  

class MyModel(Model):
    def __init__(self,classes):
        super(MyModel, self).__init__()
        self.aug1 = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.5),
        layers.experimental.preprocessing.RandomZoom(0.5),
        layers.experimental.preprocessing.RandomFlip(),
        layers.experimental.preprocessing.RandomTranslation(height_factor = 0.2,width_factor=0.2)
    ]
)
        self.conv1 = layers.Conv2D(64, (7, 7), padding="same",strides=2)
        self.bn1 = layers.BatchNormalization(axis=-1)
        self.act1= layers.Activation("relu")
        self.maxpool = layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same')
        self.residual_block1 = residual_block(64,(3,3))
        self.residual_block2 = residual_block(64,(3,3))
        self.skip_block1 = skip_block(128,(3,3))
        self.residual_block3 = residual_block(128,(3,3))
        self.skip_block2 = skip_block(256,(3,3))
        self.residual_block4 = residual_block(256,(3,3))
        self.skip_block3 = skip_block(512,(3,3))
        self.residual_block5 = residual_block(512,(3,3))
        self.avg1 = layers.AveragePooling2D(pool_size=(2,2))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(1000)
        self.act5 = layers.Activation("relu")
        self.dense3 = layers.Dense(classes)
        self.softmax = layers.Activation("softmax")
 


    def call(self,inputs):
        x = self.aug1(inputs)
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.skip_block1(x)
        x = self.residual_block3(x)    
        x = self.skip_block2(x)
        x = self.residual_block4(x,)  
        x = self.skip_block3(x)
        x = self.residual_block5(x)  
        x = self.avg1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.act5(x)
        x = self.dense3(x)
        x = self.softmax(x)
        return x
    




train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset/Train',
    target_size=(124, 124),
    batch_size=8,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    'dataset/Test',
    target_size=(124, 124),
    batch_size=8,
    class_mode='categorical')




model = MyModel(43)
print("[INFO] training network...")
epochs = 10
opt = keras.optimizers.Adagrad(learning_rate=0.01,initial_accumulator_value=0.1,epsilon=1e-7)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
model.fit_generator(train_generator,epochs=epochs,callbacks=my_callbacks, validation_data=train_generator,validation_steps=1,verbose=1)



# val_data = pd.read_csv("val_data.csv")
# X_test = val_data['path']
# X_test = val_data_generator(X_test)
# Y_test = val_data['id']
# Y_test = keras.utils.to_categorical(Y_test,num_classes=200)
# print("model final accuracy on test data")
# model.evaluate(X_test,Y_test)
