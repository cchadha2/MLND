"""
Create bottleneck features to train fully connected layers.

Create bottleneck features using pre-trained VGG-16 model. These features
are used to train fully connected layer(s) in train_model.py.
"""
from time import time

from keras.applications.vgg16 import VGG16

def bottleneck_features(X_train, X_valid, X_test):
    """
    Create bottleneck features to train fully connected layers.

    Create bottleneck features using pre-trained VGG-16 model. These features
    are used to train fully connected layer(s) in train_model.py.
    """
    ## Augment images to expand dataset.
    #datagen = image.ImageDataGenerator(rotation_range=40, width_shift_range=0.2, 
    #                                   height_shift_range=0.2, rescale=1./255, 
    #                                   shear_range=0.2, zoom_range=0.2, 
    #                                   horizontal_flip=True, fill_mode='nearest')
    
    # Initialise VGG16 model.
    model = VGG16(include_top=False, weights='imagenet', 
                  input_shape=None, pooling=None)
    
    # Output summary of model architecture.
    model.summary()
    
    # Create bottleneck features from X_train, X_valid, and X_test.
    timer_start = time()
    X_train_bottle = model.predict(X_train)
    timer_end = time()
    print('Producing bottleneck features from training images took: {}s'.format(timer_end-timer_start))
    
    timer_start = time()
    X_valid_bottle = model.predict(X_valid)
    timer_end = time()
    print('Producing bottleneck features from validation images took: {}s'.format(timer_end-timer_start))
    
    timer_start = time()
    X_test_bottle = model.predict(X_test)
    timer_end = time()
    print('Producing bottleneck features from testing images took: {}s'.format(timer_end-timer_start))
    
    return X_train_bottle, X_valid_bottle, X_test_bottle




