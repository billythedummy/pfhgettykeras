''' 
Author: Preston Lee 
Data generator for plate segmentation -- creates batches of plate and plate mask training data
'''
# Basic Libraries
import keras
import numpy as np
import random
# Image/Video Manipulation
from PIL import Image
from resizeimage import resizeimage as ri
from skimage.color import rgb2gray
from scipy import ndimage
# Detect memory leaks
from mem_top import mem_top
import gc
class DataGenerator(keras.utils.Sequence):

    def __init__(self, x_train, y_train, batch_size=10, dim=(32, 32, 3), shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.x_train = x_train
        self.y_train = y_train
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(len(self.y_train) // self.batch_size)

    def __getitem__(self, index):
        # Generate indexes of the batch from random sampling
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        x_list, y_list = [], []
        # Generate data
        for i in indexes:
            x, y = self.__data_generation(self.x_train, self.y_train, i)
            x_list.append(x)
            y_list.append(y)
        return np.vstack(tuple(x_list)), np.vstack(tuple(y_list))

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.x_train))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, x_train, y_train, f):
        rotation = random.uniform(-2, 2)
        cur_frame_raw = x_train[f]
        cur_frame_raw = self.resize_image(
            cur_frame_raw, self.dim[1], self.dim[0], rotation)

        mask_frame = None
        if f > 0:
            mask_frame = y_train[f-1]
            mask_frame = self.resize_image(
                mask_frame, self.dim[1], self.dim[0], rotation)
            mask_frame = rgb2gray(mask_frame)
        else:
            mask_frame = np.zeros(
                shape=(self.dim[0], self.dim[1], 1))

        # Add mask as another channel to current frame data to be passed into neural net
        cur_frame = np.dstack((cur_frame_raw, mask_frame))

        # Keep a reference to current frame data with blank previous mask
        cur_frame_raw = np.dstack(
            (cur_frame_raw, np.zeros(shape=(self.dim[0], self.dim[1], 1))))

        # cur_frame = np.expand_dims(cur_frame, 0)
        # cur_frame_raw = np.expand_dims(cur_frame_raw, 0)

        compare_frame = y_train[f]
        compare_frame = self.resize_image(
            compare_frame, self.dim[1], self.dim[0], rotation)
        compare_frame = rgb2gray(compare_frame)
        compare_frame = np.expand_dims(compare_frame, -1)
        # compare_frame = np.expand_dims(compare_frame, 0)
        return np.stack((cur_frame, cur_frame_raw)), np.stack((compare_frame, compare_frame))

    # Resize image, keeping all of image with black bars filling rest
    def resize_image(self, im, width, height, rotation, fill_color=(0, 0, 0)):
        im = Image.fromarray(im.astype('uint8'))        
        new_im = ri.resize_contain(im, (width, height), bg_color=fill_color)      
        new_im = new_im.rotate(rotation)  
        return np.asarray(new_im)
