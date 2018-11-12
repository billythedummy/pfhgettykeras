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
import pims
# Utility
# import matplotlib.pyplot as plt


class DataGeneratorFactory():
    def __init__(self, x_train_path, y_train_path, batch_size, dim, color_dict, shuffle=True, validation_split=0.1):
        x_train = pims.Video(x_train_path)
        y_train = pims.Video(y_train_path)
        indices = np.arange(len(y_train))
        np.random.shuffle(indices)
        val_length = int(len(y_train) * validation_split)
        validation = indices[:val_length]
        indices = indices[val_length:]

        # pims Videos cannot exist in two generators at once. To get around this,
        # construct two versions of x_train and y_train.
        x_train_copy = pims.Video(x_train_path)
        y_train_copy = pims.Video(y_train_path)
        self.datagen = DataGenerator(x_train, y_train, batch_size, dim, color_dict,
                                     shuffle=shuffle, indices=indices)
        self.validation_datagen = DataGenerator(x_train_copy, y_train_copy, batch_size,
                                                dim, color_dict, shuffle=shuffle,
                                                indices=validation, distort_images=False)


class DataGenerator(keras.utils.Sequence):
    def __init__(self, x_train, y_train, batch_size, dim, color_dict,
                 indices=None, shuffle=True, distort_images=True):
        'Initialization'
        # colour map
        self.color_dict = color_dict
        self.dim = dim
        self.batch_size = batch_size
        self.x_train = x_train
        self.y_train = y_train
        self.shuffle = shuffle
        self.distort_images = distort_images
        if(indices is None):
            self._indices = np.arange(len(y_train))
        else:
            self._indices = indices
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(len(self._indices) // self.batch_size)

    def __getitem__(self, index):
        # Generate indexes of the batch from random sampling
        indexes = self._indices[index *
                                self.batch_size:(index+1)*self.batch_size]
        x_list, y_list = [], []
        # Generate data
        for i in indexes:
            x, y = self.__data_generation(self.x_train, self.y_train, i)
            x_list.append(x)
            y_list.append(y)
        return np.stack(tuple(x_list)), np.stack(tuple(y_list))

    def on_epoch_end(self):
        # Updates indexes after each epoch
        if self.shuffle == True:
            np.random.shuffle(self._indices)

    def __data_generation(self, x_train, y_train, f):
        rotation = random.uniform(-30, 30)
        cur_frame_raw = x_train[f]
        cur_frame_raw = self.resize_image(
            cur_frame_raw, self.dim[1], self.dim[0], rotation, enhance=self.distort_images, resample=Image.LANCZOS)
        cur_frame_raw = cur_frame_raw / 255.0
        mask_frame = None
        if f > 0 and random.random() > 0.15:
            mask_frame = y_train[f-1]
            mask_frame = self.resize_image(
                mask_frame, self.dim[1], self.dim[0], rotation)
            mask_frame = self.rgb2onehot_distorted(
                mask_frame, self.dim[1], self.dim[0])
        else:
            mask_frame = np.zeros(
                shape=(self.dim[0], self.dim[1], len(self.color_dict)))

        # Add mask as another channel to current frame data to be passed into neural net
        cur_frame = np.dstack((cur_frame_raw, mask_frame))

        compare_frame = y_train[f]
        compare_frame = self.resize_image(
            compare_frame, self.dim[1], self.dim[0], rotation)
        compare_frame = self.rgb2onehot(compare_frame)
        # compare_frame = np.reshape(
        #     compare_frame, (self.dim[1] * self.dim[0], len(self.color_dict)))
        compare_frame = np.argmax(compare_frame, axis=-1)
        # compare_frame = np.expand_dims(compare_frame, 0)
        return cur_frame, compare_frame

    # Resize/augment image, keeping all of image with black bars filling rest
    def resize_image(self, im, width, height, rotation, fill_color=(0, 0, 0), enhance=False, resample=Image.NEAREST):
        im = Image.fromarray(im.astype('uint8'))
        new_im = ri.resize_contain(
            im, (width, height), bg_color=fill_color, resample=resample)

        new_im = new_im.rotate(rotation, resample=Image.NEAREST)
        if enhance:
            # PIL Image enhancements
            new_im = ImageEnhance.Contrast(
                new_im).enhance(random.uniform(0.5, 1.5))
            new_im = ImageEnhance.Brightness(
                new_im).enhance(random.uniform(0.5, 1.5))
            new_im = ImageEnhance.Sharpness(
                new_im).enhance(random.uniform(0.5, 1.5))
            new_im = ImageEnhance.Color(new_im).enhance(
                random.uniform(0.5, 1.5))

        im_array = np.asarray(new_im)
        if enhance:
            # Scikit Image enhancements
            if random.randint(0, 1):
                im_array = sk.util.random_noise(im_array, mode="s&p")

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

    def rgb2onehot_distorted(self, rgb_arr, width, height):
        shiftx = math.pow(random.uniform(-1, 1), 5) * width
        shifty = math.pow(random.uniform(-1, 1), 5) * height
        shift = (int(shifty), int(shiftx))
        noisiness = math.pow(random.uniform(0, 0.8), 2)
        onehot = self.rgb2onehot(rgb_arr)
        noise = np.random.rand(
            self.dim[0], self.dim[1], len(self.color_dict)) * noisiness
        return np.roll(np.abs(onehot - noise), shift, (0, 1))
