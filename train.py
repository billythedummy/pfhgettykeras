''' 
Author: Preston Lee 
Training loop for ARUW plate segmentation
'''
# Basic Libraries
import numpy as np
import tensorflow as tf
import sklearn.metrics as skmetrics
# Keras
import keras
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
# Visualization
from resizeimage import resizeimage as ri
import matplotlib.pyplot as plt
# Custom Libraries
from data_generator import DataGenerator, DataGeneratorFactory
from clr import CyclicLR
from lovasz_losses import lovasz_softmax
# Model
import sys
sys.path.append('./model')
from model import SegmentationNetwork

def onehot2rgb(onehot, color_dict):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros(onehot.shape[:2]+(3,))
    for k in color_dict.keys():
        output[single_layer == k] = color_dict[k]
    return np.uint8(output)

def keras_lovasz_softmax(labels,probas):
    return lovasz_softmax(probas, labels)

IMAGE_INPUT_WIDTH = 1024
IMAGE_INPUT_HEIGHT = 512
# Load a model from folder
LOAD_MODEL = False
LOAD_MODEL_PATH = "./Models/PlateSegmentation/weights-01.hdf5"
color_dict = {0: (0,   0, 0),  # 0: Background
              1: (255, 0, 0),  # 1: Red
              2: (0, 0, 255),  # 2: Blue
              }
print("Initializing...")
# Create data generator
datagen = DataGeneratorFactory(
    x_train_path="./Training/newtrainingdata.mov",
    y_train_path="./Training/newtraininglabels.mov", batch_size=6,
    dim=(IMAGE_INPUT_HEIGHT, IMAGE_INPUT_WIDTH, 3), shuffle=True,
    color_dict=color_dict, validation_split=0.05)
train_data = datagen.datagen
val_data = datagen.validation_datagen
print("Data found!")

model = SegmentationNetwork(
    layers=4, start_filters=16, squeeze_factor=8, num_classes=len(color_dict),
    shape=(IMAGE_INPUT_HEIGHT, IMAGE_INPUT_WIDTH, 3)).create_model()
print("Model built!")
if(LOAD_MODEL):
    print("Resuming model from {}".format(LOAD_MODEL_PATH))
    model.load_weights(LOAD_MODEL_PATH)

sgd = optimizers.SGD(lr=0.0)
print("Metrics initialized!")
model.compile(optimizer=sgd, loss=keras_lovasz_softmax)
print("Model compiled!")
# model = UNet(input_size = (IMAGE_INPUT_HEIGHT, IMAGE_INPUT_WIDTH, 4)).unet()


cyclical = CyclicLR(base_lr=0.0001, max_lr=1,
                    step_size=train_data.__len__() * 2, mode="triangular2")
checkpoint = ModelCheckpoint(
    "./Models/PlateSegmentation/weights-{epoch:02d}.hdf5", verbose=1, period=1)


def train(data_generator, val_generator, callbacks):
    return model.fit_generator(generator=data_generator, validation_data=val_generator,
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


# predictions = model.predict(train_data.__getitem__(0)[0])
# for pred in predictions:
#     pred_rgb = np.reshape(pred, (IMAGE_INPUT_HEIGHT, IMAGE_INPUT_WIDTH, len(color_dict)))
#     pred_rgb = onehot2rgb(pred_rgb, color_dict)
#     plt.imshow(pred_rgb)
#     plt.show()


history = train(data_generator=train_data, val_generator=val_data,
                callbacks=[cyclical, checkpoint])
plt.plot(cyclical.history['lr'], cyclical.history['loss'])
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.show()

# x_test = pims.Video("./Training/original.mov")
# test(x_test)
