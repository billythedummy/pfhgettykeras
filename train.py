''' 
Author: Preston Lee 
Training loop for ARUW plate segmentation
'''
# Basic Libraries
import numpy as np
import tensorflow as tf
# Keras
import keras.backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.backend import eval
from keras.utils.np_utils import to_categorical
# Image/Video Manipulation
import skimage
from skimage.color import rgb2gray
import skimage.io as ski
from PIL import Image
import pims
# Visualization
from resizeimage import resizeimage as ri
import matplotlib.pyplot as plt
# Custom Libraries
from data_generator import DataGenerator
from model import SegmentationNetwork
from clr import SGDRScheduler
from unet import UNet
# Accommodate large batch size
import sys
sys.setrecursionlimit(10000)

IMAGE_INPUT_WIDTH = 512
IMAGE_INPUT_HEIGHT = 256
# Load a model from folder
LOAD_MODEL = False
LOAD_MODEL_PATH = "./Models/PlateSegmentation/weights-01.hdf5"




model = SegmentationNetwork(layers=5, start_filters=16).create_model()
if(LOAD_MODEL):
    print("Resuming model from {}".format(LOAD_MODEL_PATH))
    model.load_weights(LOAD_MODEL_PATH)

sgd = optimizers.SGD(lr=0.0)
model.compile(optimizer=sgd, loss="binary_crossentropy")
# model = UNet(input_size = (IMAGE_INPUT_HEIGHT, IMAGE_INPUT_WIDTH, 4)).unet()


cyclical = SGDRScheduler(min_lr=0.00000001, max_lr=0.0005,
                         steps_per_epoch=122, lr_decay=0.8, cycle_length=1)
checkpoint = ModelCheckpoint(
    "./Models/PlateSegmentation/weights-{epoch:02d}.hdf5", verbose=1, period=1)


# Create data generator
datagen = DataGenerator(x_train=pims.Video("./Training/fulloriginalcorrected.mov"),
                        y_train=pims.Video("./Training/fullmaskcorrected.mov"), batch_size=16,
                        dim=(IMAGE_INPUT_HEIGHT, IMAGE_INPUT_WIDTH, 3), shuffle=True)


def train(data_generator, callbacks):
    model.fit_generator(generator=data_generator,
                        callbacks=callbacks, epochs=50, verbose=1)


def resize_image(im, width, height, fill_color=(0, 0, 0)):
    im = Image.fromarray(im.astype('uint8'))
    new_im = ri.resize_contain(im, (width, height), bg_color=fill_color)
    return np.asarray(new_im)


def format_image_for_save(im, remove_batch=True):
    if(remove_batch):
        image = np.squeeze(im, 0)
        image = np.squeeze(image, -1)
    else:
        image = np.squeeze(im, -1)
    return image


def test(x_test):
    mask = np.zeros(shape=(IMAGE_INPUT_HEIGHT, IMAGE_INPUT_WIDTH, 1))
    for i in range(0, len(x_test)):
        cur_frame = x_test[i]
        cur_frame = resize_image(
            cur_frame, IMAGE_INPUT_WIDTH, IMAGE_INPUT_HEIGHT)
        cur_frame = np.dstack((cur_frame, mask))
        cur_frame = np.expand_dims(cur_frame, 0)

        mask = model.predict_on_batch(cur_frame)
        mask = np.squeeze(mask, 0)
        ski.imsave("./TestOutput/segmentation" +
                   str(i)+".png", np.squeeze(mask, -1))


train(data_generator=datagen, callbacks=[cyclical, checkpoint])
x_test = pims.Video("./Training/original.mov")
test(x_test)
