"""
Process images for skin cancer detection by VGG-16.

Loads images and converts to numpy arrays for processing by VGG-16 model.
Model has been pre-trained on ImageNet data with preparation of data for 
transfer learning.
"""

import os
from time import time
import logging

from keras.preprocessing import image
from keras.utils import to_categorical
import numpy as np

__name__ = 'preprocessing.py'

# Initialise logger.
logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger(__name__)


def load_images(melanoma_train_path, melanoma_valid_path, melanoma_test_path,
                nevus_train_path, nevus_valid_path, nevus_test_path,
                seborrheic_train_path, seborrheic_valid_path,
                seborrheic_test_path):
    """
    Create numpy arrays from skin cancer images.
    
    Loads skin cancer images from paths passed as arguments, resizes for
    VGG-16, and converts to numpy arrays.
    """
    # Initialise lists to store image arrays.
    melanoma_train = []
    melanoma_valid = []
    melanoma_test = []
    nevus_train = []
    nevus_valid = []
    nevus_test = []
    seborrheic_train = []
    seborrheic_valid = []
    seborrheic_test = []
    
    # Create list of files in paths.
    melanoma_train_files = [f for f in os.listdir(melanoma_train_path) if \
                            os.path.isfile(os.path.join(melanoma_train_path, f))]
    
    melanoma_valid_files = [f for f in os.listdir(melanoma_valid_path) if \
                            os.path.isfile(os.path.join(melanoma_valid_path, f))]
    
    melanoma_test_files = [f for f in os.listdir(melanoma_test_path) if \
                           os.path.isfile(os.path.join(melanoma_test_path, f))]
    
    nevus_train_files = [f for f in os.listdir(nevus_train_path) if \
                         os.path.isfile(os.path.join(nevus_train_path, f))]
    
    nevus_valid_files = [f for f in os.listdir(nevus_valid_path) if \
                         os.path.isfile(os.path.join(nevus_valid_path, f))]
    
    nevus_test_files = [f for f in os.listdir(nevus_test_path) if \
                        os.path.isfile(os.path.join(nevus_test_path, f))]
    
    seborrheic_train_files = [f for f in os.listdir(seborrheic_train_path) if \
                              os.path.isfile(os.path.join(seborrheic_train_path, f))]
    
    seborrheic_valid_files = [f for f in os.listdir(seborrheic_valid_path) if \
                              os.path.isfile(os.path.join(seborrheic_valid_path, f))]
    
    seborrheic_test_files = [f for f in os.listdir(seborrheic_test_path) if \
                             os.path.isfile(os.path.join(seborrheic_test_path, f))]
    
    total_melanoma = (len(melanoma_train_files)
                      +len(melanoma_valid_files)
                      +len(melanoma_test_files))
    
    total_nevus = (len(nevus_train_files)
                   +len(nevus_valid_files)
                   +len(nevus_test_files))
    
    total_seborrheic = (len(seborrheic_train_files)
                        +len(seborrheic_valid_files)
                        +len(seborrheic_test_files))
    
    # Print number of images in each class for each subset.
    logger.debug("{} melanoma training images".format(len(melanoma_train_files)))
    logger.debug("{} melanoma validation images".format(len(melanoma_valid_files)))
    logger.debug("{} melanoma testing images".format(len(melanoma_test_files)))
    logger.debug("{} melanoma images in total \n".format(total_melanoma))
    
    logger.debug("{} nevus training images".format(len(nevus_train_files)))
    logger.debug("{} nevus validation images".format(len(nevus_valid_files)))
    logger.debug("{} nevus testing images".format(len(nevus_test_files)))
    logger.debug("{} nevus images in total \n".format(total_nevus))
    
    logger.debug("{} seborrheic training images".format(len(seborrheic_train_files)))
    logger.debug("{} seborrheic validation images".format(len(seborrheic_valid_files)))
    logger.debug("{} seborrheic testing images".format(len(seborrheic_test_files)))
    logger.debug("{} seborrheic images in total \n".format(total_seborrheic))
    
    # Read melanoma images into memory as array- not best way of going about this?
    timer_start = time()
    for img in melanoma_train_files:
        img_load = image.load_img('../../data/train/melanoma/' + img,
                              target_size = (244, 244))
        img_matrix = image.img_to_array(img_load)
    #    x = x.reshape((1,) + x.shape)
        melanoma_train.append(img_matrix)
        logger.debug('{}/{} melanoma training images read to list of arrays'\
                     .format(len(melanoma_train), len(melanoma_train_files)))
    timer_end = time()
    total_time = timer_end - timer_start
    logger.debug('Time to read melanoma train images into memory: {:.2f}s'.format(total_time))
    melanoma_train = np.array(melanoma_train)
    
    timer_start = time()
    for img in melanoma_valid_files:
        img_load = image.load_img('../../data/valid/melanoma/' + img,
                              target_size = (244, 244))
        img_matrix = image.img_to_array(img_load)
    #    img_matrix = img_matrix.reshape((1,) + img_matrix.shape)
        melanoma_valid.append(img_matrix)
        logger.debug('{}/{} melanoma validation images read to list of arrays'\
                     .format(len(melanoma_valid), len(melanoma_valid_files)))
    timer_end = time()
    total_time = timer_end - timer_start
    logger.debug('Time to read melanoma valid images into memory: {:.2f}s'.format(total_time))
    melanoma_valid = np.array(melanoma_valid)
    
    timer_start = time()
    for img in melanoma_test_files:
        img_load = image.load_img('../../data/test/melanoma/' + img,
                              target_size = (244, 244))
        img_matrix = image.img_to_array(img_load)
    #    img_matrix = img_matrix.reshape((1,) + img_matrix.shape)
        melanoma_test.append(img_matrix)
        logger.debug('{}/{} melanoma testing images read to list of arrays'\
                     .format(len(melanoma_test), len(melanoma_test_files)))
    timer_end = time()
    total_time = timer_end - timer_start
    logger.debug('Time to read melanoma test images into memory: {:.2f}s'.format(total_time))
    melanoma_test = np.array(melanoma_test)
    
    # Read nevus images into memory as array.
    timer_start = time()
    for img in nevus_train_files:
        img_load = image.load_img('../../data/train/nevus/' + img,
                              target_size = (244, 244))
        img_matrix = image.img_to_array(img_load)
    #    x = x.reshape((1,) + x.shape)
        nevus_train.append(img_matrix)
        logger.debug('{}/{} nevus training images read to list of arrays'\
                     .format(len(nevus_train), len(nevus_train_files)))
    timer_end = time()
    total_time = timer_end - timer_start
    logger.debug('Time to read nevus train images into memory: {:.2f}s'.format(total_time))
    logger.debug('')
    nevus_train = np.array(nevus_train)
    
    timer_start = time()
    for img in nevus_valid_files:
        img_load = image.load_img('../../data/valid/nevus/' + img,
                              target_size = (244, 244))
        img_matrix = image.img_to_array(img_load)
    #    img_matrix = img_matrix.reshape((1,) + img_matrix.shape)
        nevus_valid.append(img_matrix)
        logger.debug('{}/{} nevus validation images read to list of arrays'\
                     .format(len(nevus_valid), len(nevus_valid_files)))
    timer_end = time()
    total_time = timer_end - timer_start
    logger.debug('Time to read nevus valid images into memory: {:.2f}s'.format(total_time))
    nevus_valid = np.array(nevus_valid)
    
    timer_start = time()
    for img in nevus_test_files:
        img_load = image.load_img('../../data/test/nevus/' + img,
                              target_size = (244, 244))
        img_matrix = image.img_to_array(img_load)
    #    img_matrix = img_matrix.reshape((1,) + img_matrix.shape)
        nevus_test.append(img_matrix)
        logger.debug('{}/{} nevus testing images read to list of arrays'\
                     .format(len(nevus_test), len(nevus_test_files)))
    timer_end = time()
    total_time = timer_end - timer_start
    logger.debug('Time to read nevus test images into memory: {:.2f}s'.format(total_time))
    nevus_test = np.array(nevus_test)
    
    # Read seborrheic images into memory as array.
    timer_start = time()
    for img in seborrheic_train_files:
        img_load = image.load_img('../../data/train/seborrheic_keratosis/' + img,
                              target_size = (244, 244))
        img_matrix = image.img_to_array(img_load)
    #    x = x.reshape((1,) + x.shape)
        seborrheic_train.append(img_matrix)
        logger.debug('{}/{} seborrheic training images read to list of arrays'\
                     .format(len(seborrheic_train), len(seborrheic_train_files)))
    timer_end = time()
    total_time = timer_end - timer_start
    logger.debug('Time to read seborrheic train images into memory: {:.2f}s'.format(total_time))
    logger.debug('')
    seborrheic_train = np.array(seborrheic_train)
    
    timer_start = time()
    for img in seborrheic_valid_files:
        img_load = image.load_img('../../data/valid/seborrheic_keratosis/' + img,
                              target_size = (244, 244))
        img_matrix = image.img_to_array(img_load)
    #    img_matrix = img_matrix.reshape((1,) + img_matrix.shape)
        seborrheic_valid.append(img_matrix)
        logger.debug('{}/{} seborrheic validation images read to list of arrays'\
                     .format(len(seborrheic_valid), len(seborrheic_valid_files)))
    timer_end = time()
    total_time = timer_end - timer_start
    logger.debug('Time to read seborrheic valid images into memory: {:.2f}s'.format(total_time))
    seborrheic_valid = np.array(seborrheic_valid)
    
    timer_start = time()
    for img in seborrheic_test_files:
        img_load = image.load_img('../../data/test/seborrheic_keratosis/' + img,
                              target_size = (244, 244))
        img_matrix = image.img_to_array(img_load)
    #    img_matrix = img_matrix.reshape((1,) + img_matrix.shape)
        seborrheic_test.append(img_matrix)
        logger.debug('{}/{} seborrheic testing images read to list of arrays'\
                     .format(len(seborrheic_test), len(seborrheic_test_files)))
    timer_end = time()
    total_time = timer_end - timer_start
    logger.debug('Time to read seborrheic test images into memory: {:.2f}s'.format(total_time))
    seborrheic_test = np.array(seborrheic_test)
    
    # Create numpy array of training arrays.
    X_train =  np.append(melanoma_train, nevus_train, axis = 0)
    X_train = np.append(X_train, seborrheic_train, axis = 0)
    X_train = X_train.astype('float32')/255
    
    # Create numpy array of training arrays.
    X_valid =  np.append(melanoma_valid, nevus_valid, axis = 0)
    X_valid = np.append(X_valid, seborrheic_valid, axis = 0)
    X_valid = X_valid.astype('float32')/255
    
    # Create numpy array of training arrays.
    X_test =  np.append(melanoma_test, nevus_test, axis = 0)
    X_test = np.append(X_test, seborrheic_test, axis = 0)
    X_test = X_test.astype('float32')/255
    
    # Create arrays of labels and one-hot encode.
    y_train = np.zeros(len(melanoma_train))
    y_train = np.append(y_train, np.ones(len(nevus_train)))
    a = np.array(range(len(seborrheic_train)))
    a.fill(2)
    y_train = np.append(y_train, a)
    y_train = to_categorical(y_train, 3)

    y_valid = np.zeros(len(melanoma_valid))
    y_valid = np.append(y_valid, np.ones(len(nevus_valid)))
    b = np.array(range(len(seborrheic_valid)))
    b.fill(2)
    y_valid = np.append(y_valid, b)
    y_valid = to_categorical(y_valid, 3)

    y_test = np.zeros(len(melanoma_test))
    y_test = np.append(y_test, np.ones(len(nevus_test)))
    c = np.array(range(len(seborrheic_test)))
    c.fill(2)
    y_test = np.append(y_test, c)
    y_test = to_categorical(y_test, 3)
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test
            