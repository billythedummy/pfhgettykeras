''' 
Author: Preston Lee 
Data generator for plate segmentation -- creates batches of plate and plate mask training data
'''
# Basic Libraries
import tensorflow as tf
import keras
import numpy as np
import random
import math
# Image/Video Manipulation
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
from resizeimage import resizeimage as ri
from scipy import ndimage
import skimage as sk
from skimage import util
# Utility
import matplotlib.pyplot as plt


class DataGenerator(keras.utils.Sequence):

    def __init__(self, x_train, y_train, batch_size=10, dim=(32, 32, 3), color_dict=None, shuffle=True):
        'Initialization'
        # colour map
        self.color_dict = color_dict
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
        return np.stack(tuple(x_list)), np.stack(tuple(y_list))

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.y_train))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, x_train, y_train, f):
        rotation = random.uniform(-30, 30)
        cur_frame_raw = x_train[f]
        cur_frame_raw = self.resize_image(
            cur_frame_raw, self.dim[1], self.dim[0], rotation, enhance=True, resample=Image.LANCZOS)

        mask_frame = None
        if f > 0 and random.random() > 0.15:
            mask_frame = y_train[f-1]
            mask_frame = self.resize_image(
                mask_frame, self.dim[1], self.dim[0], rotation + random.uniform(-4, 4))
            mask_frame = self.rgb2onehot_distorted(mask_frame)
        else:
            mask_frame = np.zeros(
                shape=(self.dim[0], self.dim[1], len(self.color_dict)))

        # Add mask as another channel to current frame data to be passed into neural net
        cur_frame = np.dstack((cur_frame_raw, mask_frame))

        compare_frame = y_train[f]
        compare_frame = self.resize_image(
            compare_frame, self.dim[1], self.dim[0], rotation)
        compare_frame = self.rgb2onehot(compare_frame)
        compare_frame = np.reshape(compare_frame, (self.dim[0] * self.dim[1], len(self.color_dict)))
        # compare_frame = np.expand_dims(compare_frame, 0)
        return cur_frame, compare_frame

    # Resize/augment image, keeping all of image with black bars filling rest
    def resize_image(self, im, width, height, rotation, fill_color=(0, 0, 0), enhance=False, resample=Image.NEAREST):
        im = Image.fromarray(im.astype('uint8'))
        new_im = ri.resize_contain(
            im, (width, height), bg_color=fill_color, resample=resample)

        new_im = new_im.rotate(rotation, resample=Image.NEAREST)
        if enhance:
            # Image enhancements
            new_im = ImageEnhance.Contrast(
                new_im).enhance(random.uniform(0.7, 1.4))
            new_im = ImageEnhance.Brightness(
                new_im).enhance(random.uniform(0.7, 1.4))
            new_im = ImageEnhance.Sharpness(
                new_im).enhance(random.uniform(0.7, 1.4))
        im_array = np.asarray(new_im)
        return im_array

    def rgb2onehot(self, rgb_arr):
        num_classes = len(self.color_dict)
        # shape will be original (width, height) with "num_classes" channels
        shape = rgb_arr.shape[:2]+(num_classes,)
        arr = np.zeros(shape, dtype=np.float32)

        # for each of our "num_classes" channels, each corresponding to a class...
        for i, cls in enumerate(self.color_dict):
            all_pixels = rgb_arr.reshape((-1, 3))
            matching_pixels = all_pixels == self.color_dict[i]
            # assign back to this class's channel in the output
            arr[:, :, i] = np.all(matching_pixels, axis=1).reshape(shape[:2])
        return arr

    def rgb2onehot_distorted(self, rgb_arr):
        noisiness = math.pow(random.uniform(0, 0.8), 2)
        onehot = self.rgb2onehot(rgb_arr)
        noise = np.random.rand(self.dim[0], self.dim[1], len(self.color_dict)) * noisiness
        return np.abs(onehot - noise)
