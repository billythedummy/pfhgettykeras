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
# Model
import sys
sys.path.append('./model')
from base_model import SegmentationNetwork

IMAGE_INPUT_WIDTH = 512
IMAGE_INPUT_HEIGHT = 256
LOAD_MODEL_PATH = "./Models/PlateSegmentation/weights-15.hdf5"
color_dict = {0: (0,   0, 0),  # 0: Background
              1: (255, 0, 0),  # 1: Red
              2: (0, 0, 255),  # 2: Blue
              }
K.set_learning_phase(0)
model = SegmentationNetwork(
    layers=4, start_filters=16, squeeze_factor=8, num_classes=len(color_dict),
    shape=(IMAGE_INPUT_HEIGHT, IMAGE_INPUT_WIDTH, 3)).create_model()

print("Resuming model from {}".format(LOAD_MODEL_PATH))
model.load_weights(LOAD_MODEL_PATH)

print("Model loaded!")


def resize_image(im, width, height, fill_color=(0, 0, 0)):
    new_im = Image.fromarray(im.astype('uint8'))
    new_im = ri.resize_contain(new_im, (width, height), bg_color=fill_color)
    return np.asarray(new_im)


def onehot2rgb(onehot, color_dict):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros(onehot.shape[:2]+(3,))
    for k in color_dict.keys():
        output[single_layer == k] = color_dict[k]
    return np.uint8(output)


def rgb2onehot(rgb_arr):
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]+(num_classes,)
    arr = np.zeros(shape, dtype=np.int8)
    for i, cls in enumerate(color_dict):
        arr[:, :, i] = np.all(rgb_arr.reshape(
            (-1, 3)) == color_dict[i], axis=1).reshape(shape[:2])
    return arr


def test(x_test, y_test, temporal=False):
    mask = np.zeros(shape=(IMAGE_INPUT_HEIGHT,
                           IMAGE_INPUT_WIDTH, len(color_dict)))
    for i in range(1, len(x_test)):
        if(not temporal):
            mask = np.zeros(shape=(IMAGE_INPUT_HEIGHT,
                            IMAGE_INPUT_WIDTH, len(color_dict)))
        # mask = y_test[i-1]
        # mask = resize_image(mask, IMAGE_INPUT_WIDTH, IMAGE_INPUT_HEIGHT)
        # mask = rgb2onehot(mask)
        cur_frame = x_test[i]
        cur_frame = resize_image(
            cur_frame, IMAGE_INPUT_WIDTH, IMAGE_INPUT_HEIGHT)
        cur_frame = cur_frame / 255.0
        cur_frame = np.dstack((cur_frame, mask))
        cur_frame = np.expand_dims(cur_frame, 0)

        new_mask = model.predict_on_batch(cur_frame)
        new_mask = np.squeeze(new_mask, 0)
        new_mask = np.reshape(new_mask, (IMAGE_INPUT_HEIGHT, IMAGE_INPUT_WIDTH, len(color_dict)))
        # mask = np.transpose(mask, (1, 0))
        new_mask = onehot2rgb(new_mask, color_dict)

        ski.imsave("./TestOutput/segmentation" +
                   str(i)+".png", np.copy(new_mask))
        mask = rgb2onehot(new_mask)


x_test = pims.Video("./Training/newtrainingdata.mov")
y_test = pims.Video("./Training/newtraininglabels.mov")
test(x_test, y_test)
