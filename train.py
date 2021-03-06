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
from keras.callbacks import TensorBoard
from iou import mean_iou, build_iou_for
import time
#import matplotlib.pyplot as plt
# Custom Libraries
from data_generator import DataGenerator, DataGeneratorFactory
from clr import CyclicLR
from lovasz_losses import lovasz_softmax
from lr_finder import LRFinder
# Model
import sys
sys.path.append('./model')
from model_skip import SegmentationNetwork

flags = tf.app.flags
flags.DEFINE_string('x_train', '', 'Path to the raw video (x_train_path)')
flags.DEFINE_string('y_train', '', 'Path to labelled mask video (y_train_path)')
flags.DEFINE_string('weights', '', 'Path to weights')
flags.DEFINE_boolean('from_chkpt', True, 'Resume training from checkpoint?')
flags.DEFINE_string('chkpt', '', 'Directory to save checkpoints to')
flags.DEFINE_string('lr', '1e-3', 'Learning Rate')
flags.DEFINE_string('max_lr', '5e-3', 'Max Learning Rate')

FLAGS = flags.FLAGS

def onehot2rgb(onehot, color_dict):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros(onehot.shape[:2]+(3,))
    for k in color_dict.keys():
        output[single_layer == k] = color_dict[k]
    return np.uint8(output)

IMAGE_INPUT_WIDTH = 512
IMAGE_INPUT_HEIGHT = 256
BATCH_SIZE = 6
# Load a model from folder
LOAD_MODEL = FLAGS.from_chkpt
LOAD_MODEL_PATH = FLAGS.weights
color_dict = {0: (0,   0, 0),  # 0: Background
              1: (255, 0, 0),  # 1: Red
              2: (0, 0, 255),  # 2: Blue
              }


# Losses
def keras_lovasz_softmax(labels, probas):        
    max_labels = tf.argmax(labels, axis=-1)
    return lovasz_softmax(probas, max_labels, classes="present", per_image=True)

def focal_loss(target, output, gamma=2):
    output /= K.sum(output, axis=-1, keepdims=True)
    eps = K.epsilon()
    output = K.clip(output, eps, 1. - eps)
    return -K.sum(K.pow(1. - output, gamma) * target * K.log(output),
                  axis=-1)
def weighted_pixelwise_crossentropy(class_weights):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        return -tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred), 
            class_weights))
    return loss
print("Initializing...")
# Create data generator
datagen = DataGeneratorFactory(
    x_train_path=FLAGS.x_train,
    y_train_path=FLAGS.y_train, batch_size=BATCH_SIZE,
    dim=(IMAGE_INPUT_HEIGHT, IMAGE_INPUT_WIDTH, 3), shuffle=True,
    color_dict=color_dict, validation_split=0.05)
train_data = datagen.datagen
val_data = datagen.validation_datagen
print("Data found!")

model = SegmentationNetwork(
    layers=3, start_filters=32, num_classes=len(color_dict),
    shape=(IMAGE_INPUT_HEIGHT, IMAGE_INPUT_WIDTH, 3)).create_model()
print("Model built!")
model.summary()
if(LOAD_MODEL):
    print("Resuming model from {}".format(LOAD_MODEL_PATH))
    model.load_weights(LOAD_MODEL_PATH)

sgd = optimizers.SGD(lr=0.0, momentum=.99, nesterov=True)
print("Metrics initialized!")
bla_iou = build_iou_for(0, "black_iou")
r_iou = build_iou_for(1, "red_iou")
blu_iou = build_iou_for(2, "blue_iou")
model.compile(optimizer=sgd, 
    loss=weighted_pixelwise_crossentropy([0.00418313, 0.509627837, 1.]), 
    #loss = keras_lovasz_softmax,
    sample_weight_mode="temporal", metrics=[bla_iou, r_iou, blu_iou])
print("Model compiled!")

# Callbacks
tensorboard = TensorBoard(
    log_dir="logs/{}".format(time.time()), write_graph=True, update_freq="batch")
print("Tensorboard loaded!")
# 5e-5
cyclical = CyclicLR(base_lr=float(FLAGS.lr), max_lr=float(FLAGS.max_lr),
                    step_size=train_data.__len__() * 2.5, mode="triangular2")
checkpoint = ModelCheckpoint(
    FLAGS.chkpt.rstrip("/") + "/weights-{epoch:02d}.hdf5", verbose=1, period=1)


def train(data_generator, val_generator, callbacks):
    return model.fit_generator(generator=data_generator, validation_data=val_generator,
                               callbacks=callbacks, epochs=50, verbose=1,
                                shuffle=False)


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

history = train(data_generator=train_data, val_generator=val_data,
                callbacks=[cyclical, checkpoint, tensorboard])

