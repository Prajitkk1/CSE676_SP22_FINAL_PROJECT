# -*- coding: utf-8 -*-

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from matplotlib import pyplot as plt
import copy

from tqdm import tqdm
from PIL import Image
from keras import Input
from keras.layers import Dense, Reshape, LeakyReLU, Conv2D, Conv2DTranspose, Flatten, Dropout
from keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model

#generator
def create_generator():
    gen_input = Input(shape=(16, ))
    
    x = Dense(128 * 16 * 16)(gen_input)
    
    x = LeakyReLU()(x)
    x = Reshape((16, 16, 128))(x)

    x = Conv2D(256, 5, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)
    print("shape")
    print(x.shape)
    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2D(512, 5, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(512, 5, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(3, 7, activation='tanh', padding='same')(x)

    generator = Model(gen_input, x)
    return generator



#descriminator
def create_discriminator(HEIGHT, WIDTH):
    disc_input = Input(shape=(HEIGHT, WIDTH, 3))

    x = Conv2D(256, 3)(disc_input)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Flatten()(x)
    x = Dropout(0.4)(x)

    x = Dense(1, activation='sigmoid')(x)
    discriminator = Model(disc_input, x)

    optimizer = RMSprop(
        lr=.0001,
        clipvalue=1.0,
        decay=1e-8
    )

    discriminator.compile(
        optimizer=optimizer,
        loss='binary_crossentropy'
    )

    return discriminator


            
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
    tf.keras.callbacks.ModelCheckpoint(filepath='best_model_classification_withGAN/best_model_resnet_1',save_best_only=True,save_format='tf'),
    tf.keras.callbacks.TensorBoard(log_dir='.\logstf\classification_withGAN'),
    tf.keras.callbacks.CSVLogger("model_history_classification_withGAN.csv",append=True)
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