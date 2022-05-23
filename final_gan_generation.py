# -*- coding: utf-8 -*-


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from matplotlib import pyplot as plt

from tqdm import tqdm
from PIL import Image
from keras import Input
from keras.layers import Dense, Reshape, LeakyReLU, Conv2D, Conv2DTranspose, Flatten, Dropout
from keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from keras.models import load_model
import tensorflow as tf
from gan_utils import create_generator, create_discriminator
import time
from PIL import Image as Img

###################count number of images in each class##################
pic_dir = 'dataset/Train/'

cnt = len(os.listdir(pic_dir))
lst = list()
for i in range(cnt):
    PIC_DIR = 'dataset/Train/' + str(i)+'/'
    lst.append(len(os.listdir(PIC_DIR)))
    
max_images = 600
iters =800
batch_size = 4
CONTROL_SIZE_SQRT = 3
LATENT_DIM = 16

control_vectors = np.random.normal(size=(CONTROL_SIZE_SQRT**2, LATENT_DIM)) / 2

for i in range(cnt):
    if lst[i] < max_images:
        PIC_DIR = 'dataset/Train/'+ str(i)+'/'
        reqd_images = max_images - lst[i]
        IMAGES_COUNT = lst[i]
        LATENT_DIM = 16
        WIDTH = 128
        HEIGHT = 128
        image_size_expected = (WIDTH, HEIGHT)
        if lst[i]<300:
            iters = 1200
        elif lst[i]>300:
            iters = 700
        images = [] 
        for pic_file in tqdm(os.listdir(PIC_DIR)[:IMAGES_COUNT]):
            pic = Image.open(PIC_DIR + pic_file)
            if pic.size[0]<64:
                continue
            pic= pic.resize(image_size_expected)
            images.append(np.uint8(pic))
            
            
        images = np.array(images) / 255

        CHANNELS = 3
        
        generator = create_generator()
        discriminator = create_discriminator(HEIGHT,WIDTH)
        discriminator.trainable = False
        
        gan_input = Input(shape=(LATENT_DIM, ))
        gen = generator(gan_input)
        gan_output = discriminator(gen)
        gan = Model(gan_input, gan_output)
        optimizer = RMSprop(lr=.0001, clipvalue=1.0, decay=1e-8)
        gan.compile(optimizer=optimizer, loss='binary_crossentropy')
    
        start = 0
        d_losses = []
        a_losses = []
        images_saved = 0
        best_loss = np.inf
    
        for step in range(iters):
            latent_vectors = np.random.normal(size=(batch_size, LATENT_DIM))
            generated = generator.predict(latent_vectors)
        
            real = images[start:start + batch_size]
            combined_images = np.concatenate([generated, real])
        
            labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
            labels += .05 * np.random.random(labels.shape)
        
            d_loss = discriminator.train_on_batch(combined_images, labels)
            if d_loss<best_loss:
                generator.save_weights("generator_best_model"+str(i)+".h5")
                best_loss=  d_loss
            d_losses.append(d_loss)
        
            latent_vectors = np.random.normal(size=(batch_size, LATENT_DIM))
            misleading_targets = np.zeros((batch_size, 1))
        
            a_loss = gan.train_on_batch(latent_vectors, misleading_targets)
            a_losses.append(a_loss)
        
            start += batch_size
            if start > images.shape[0] - batch_size:
                start = 0
        
            if step % 50 == 49:
                #gan.save_weights('/gan.h5')    
                print('%d/%d: d_loss: %.4f,  a_loss: %.4f. ' % (step + 1, iters, d_loss, a_loss))
        RES_DIR = 'class' + str(i)
        if not os.path.isdir(RES_DIR):
            os.mkdir(RES_DIR)
        
        model_best = create_generator()
        model_best.load_weights("generator_best_model"+str(i)+".h5")
        for j in range(reqd_images):
            control_vectors = np.random.normal(size=(1, LATENT_DIM)) / 2
            control_generated = generator.predict(control_vectors)
            im = Img.fromarray(np.uint8(control_generated[0] *255))
            FILE_PATH = '%s/generated_%d.png'
            im.save(FILE_PATH % (RES_DIR, j))
    
    
