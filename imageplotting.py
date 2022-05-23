# -*- coding: utf-8 -*-


#import libraries
import cv2
from matplotlib import pyplot as plt
  
# create figure
fig = plt.figure(figsize=(10, 7))
plt.title("Generated Images")
plt.axis('off')

# setting values to rows and column variables
rows = 2
columns = 2
  
# reading images
Image1 = cv2.imread('class0/generated_3.png')
Image2 = cv2.imread('class19/generated_0.png')
Image3 = cv2.imread('class21/generated_0.png')
Image4 = cv2.imread('class29/generated_0.png')
  
# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)
  
# showing image
plt.imshow(Image1)
plt.axis('off')

  
# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
  
# showing image
plt.imshow(Image2)
plt.axis('off')

  
# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)
  
# showing image
plt.imshow(Image3)
plt.axis('off')

  
# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 4)
plt.axis('off')

# showing image

plt.imshow(Image4)
