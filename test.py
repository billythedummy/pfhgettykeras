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
IMAGE_INPUT_WIDTH = 512
IMAGE_INPUT_HEIGHT = 256
LOAD_MODEL_PATH = "./Models/PlateSegmentation/weights-50.hdf5"
model = SegmentationNetwork(layers=5, start_filters=16).create_model()

print("Resuming model from {}".format(LOAD_MODEL_PATH))
model.load_weights(LOAD_MODEL_PATH)
def resize_image(im, width, height, fill_color=(0, 0, 0)):
    im = Image.fromarray(im.astype('uint8'))
    new_im = ri.resize_contain(im, (width, height), bg_color=fill_color)
    return np.asarray(new_im)
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

x_test = pims.Video("./Test/20181006_171534.mp4")
test(x_test)
