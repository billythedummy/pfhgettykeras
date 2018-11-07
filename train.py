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
from data_generator import DataGenerator, DataGeneratorFactory
from model import SegmentationNetwork
from clr import CyclicLR


def onehot2rgb(onehot, color_dict):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros(onehot.shape[:2]+(3,))
    for k in color_dict.keys():
        output[single_layer == k] = color_dict[k]
    return np.uint8(output)


IMAGE_INPUT_WIDTH = 2048
IMAGE_INPUT_HEIGHT = 1024
# Load a model from folder
LOAD_MODEL = False
LOAD_MODEL_PATH = "./Models/PlateSegmentation/weights-01.hdf5"
color_dict = {0: (0,   0, 0),  # 0: Background
              1: (255, 0, 0),  # 1: Red
              2: (0, 0, 255),  # 2: Blue
              }
# Create data generator
datagen = DataGeneratorFactory(
    x_train=pims.Video("./Training/newtrainingdata.mov"),
    y_train=pims.Video("./Training/newtraininglabels.mov"), batch_size=4,
    dim=(IMAGE_INPUT_HEIGHT, IMAGE_INPUT_WIDTH, 3), shuffle=True,
    color_dict=color_dict, validation_split=0.05)
train_data = datagen.datagen
val_data = datagen.validation_datagen


model = SegmentationNetwork(
    layers=4, start_filters=16, squeeze_factor=16, num_classes=len(color_dict),
    shape=(IMAGE_INPUT_HEIGHT, IMAGE_INPUT_WIDTH, 3)).create_model()

if(LOAD_MODEL):
    print("Resuming model from {}".format(LOAD_MODEL_PATH))
    model.load_weights(LOAD_MODEL_PATH)

sgd = optimizers.SGD(lr=0.0)
model.compile(optimizer=sgd, loss="categorical_crossentropy",
              metrics=['categorical_accuracy'])
# model = UNet(input_size = (IMAGE_INPUT_HEIGHT, IMAGE_INPUT_WIDTH, 4)).unet()


cyclical = CyclicLR(base_lr=0.000001, max_lr=0.01,
                    step_size=train_data.__len__() * 2, mode="triangular2")
checkpoint = ModelCheckpoint(
    "./Models/PlateSegmentation/weights-{epoch:02d}.hdf5", verbose=1, period=1)


def train(data_generator, val_generator, callbacks):
    return model.fit_generator(generator=data_generator, validation_data=val_generator,
                               callbacks=callbacks, epochs=100, verbose=1, use_multiprocessing=True, workers=1)


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


def rgb2onehot(rgb_arr):
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]+(num_classes,)
    arr = np.zeros(shape, dtype=np.int8)
    for i, cls in enumerate(color_dict):
        arr[:, :, i] = np.all(rgb_arr.reshape(
            (-1, 3)) == color_dict[i], axis=1).reshape(shape[:2])
    return arr


# def test(x_test):
#     mask = np.zeros(shape=(IMAGE_INPUT_HEIGHT,
#                            IMAGE_INPUT_WIDTH, len(color_dict)))
#     for i in range(0, len(x_test)):
#         cur_frame = x_test[i]
#         cur_frame = resize_image(
#             cur_frame, IMAGE_INPUT_WIDTH, IMAGE_INPUT_HEIGHT)
#         cur_frame = np.dstack((cur_frame, mask))
#         cur_frame = np.expand_dims(cur_frame, 0)

#         mask = model.predict_on_batch(cur_frame)
#         mask = np.squeeze(mask, 0)
#         mask = onehot2rgb(mask, color_dict)
#         ski.imsave("./TestOutput/segmentation" +
#                    str(i)+".png", mask)
#         mask = rgb2onehot(mask)


history = train(data_generator=train_data, val_generator=val_data,
                callbacks=[cyclical, checkpoint])
# plt.plot(cyclical.history['lr'], cyclical.history['loss'])
# plt.xlabel('Learning Rate')
# plt.ylabel('Loss')
# plt.show()

# x_test = pims.Video("./Training/original.mov")
# test(x_test)
