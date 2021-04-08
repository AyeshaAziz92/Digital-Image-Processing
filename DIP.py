#!/usr/bin/env python
# coding: utf-8

# In[1]:


#filpping an image
from PIL import Image
def image_flip(img):
    imageObject = Image.open("image.jpg")



#orginal image 
    imageObject.show()

# flipping vertically
    hori_flippedImage = imageObject.transpose(Image.FLIP_LEFT_RIGHT)


# Showing the image 
    Vert_flippedImage = imageObject.transpose(Image.FLIP_TOP_BOTTOM)
    return Vert_flippedImage.show()
img = Image.open('image.jpg')
image_flip(img)


# In[21]:


# bit plane slicing

import cv2 
import numpy as np

def image_slicing(img):

    img = cv2.imread('image.jpg', 0) 

    out = []

    for k in range(0, 7):
    # create an image for each k bit plane
        plane = np.full((img.shape[0], img.shape[1]), 2 ** k, np.uint8)
    # execute bitwise and operation
        res = cv2.bitwise_and(plane, img)
    # multiply ones (bit plane sliced) with 255 just for better visualization
        x = res * 255
    # append to the output list
        out.append(x)

    cv2.imshow("bit plane", np.hstack(out))
    return cv2.waitKey()
img = Image.open('image.jpg')
image_slicing(img)


# In[ ]:





# In[48]:


#negative-image
import cv2
import numpy as np

def image_negative(img):
    img = cv2.imread('image.jpg')

    print(img.dtype)

    img_neg = 255 - img

    cv2.imshow('negative',img_neg)
    return cv2.waitKey(0)
image_negative(img)


# In[3]:





# In[25]:



#contrast by using defination of avergae pixel difference
import numpy as np
from PIL import Image


def image_contrast(img):


    im = Image.open('image.jpg')
    im_grey = im.convert('LA') 
    width, height = im.size

    total = 0
    for i in range(0, width):
        for j in range(0, height):
            total += im_grey.getpixel((i,j))[0]

    mean = total / (width * height)
    return print(mean)
img = Image.open('image.jpg')
image_contrast(img)


# In[28]:


#power law transformation 

import numpy as np
import cv2

def image_log_trans(img):
    img = cv2.imread('image.jpg')

    gamma_two_point_two = np.array(255*(img/255)**2.2,dtype='uint8')

    gamma_point_four = np.array(255*(img/255)**0.4,dtype='uint8')

    img3 = cv2.hconcat([gamma_two_point_two,gamma_point_four])
    cv2.imshow('a2',img3)
    return cv2.waitKey(0)
img = Image.open('image.jpg')
image_log_trans(img)


# In[7]:


#contrast stretching 
import numpy as np
import cv2

def image_streching(img):
    img1 = cv2.imread('image.jpg',0)
 

    minmax_img = np.zeros((img1.shape[0],img1.shape[1]),dtype = 'uint8')

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            minmax_img[i,j] = 255*(img1[i,j]-np.min(img1))/(np.max(img1)-np.min(img1))
 

    cv2.imshow('Minmax',minmax_img)
    return cv2.waitKey(0)
img = Image.open('image.jpg')
image_streching(img)


# In[12]:


#entropy

from PIL import Image
import math
 
def image_entropy(img):
    """calculate the entropy of an image"""
    histogram = img.histogram()
    histogram_length = sum(histogram)
 
    samples_probability = [float(h) / histogram_length for h in histogram]
 
    return -sum([p * math.log(p, 2) for p in samples_probability if p != 0])
 
img = Image.open('image.jpg')
image_entropy(img)


# In[45]:


# thresholding 
import cv2  
import numpy as np  
 

def image_thres(img):
    
    image1 = cv2.imread('image.jpg')  
 
    img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) 
 
    ret, thresh1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY) 

    cv2.imshow('Thresholding', thresh1) 


    
# De-allocate any associated memory usage   
    if cv2.waitKey(0) & 0xff == 27:  
        return cv2.destroyAllWindows()  
image_thres(img)


# In[47]:





# In[ ]:




