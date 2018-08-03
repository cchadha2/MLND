"""
Classify melanoma vs. nevus & seborrheic and seborrheic vs. melanoma & nevus.

Using pre-trained VGG-16 on ImageNet data, create transfer learning model
that classifies melanoma images against nevus & seborrheic-keratosis images
and seborrheic-keratosis images against melanoma & nevus images.
"""

import os
import sys

import pandas as pd
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

sys.path.insert(0, os.path.abspath("../"))
from preprocessing.preprocessing import load_images
from bottleneck import bottleneck_features

# Define paths to image files.
melanoma_train_path = "../../data/train/melanoma"
melanoma_valid_path = "../../data/valid/melanoma"
melanoma_test_path = "../../data/test/melanoma"
nevus_train_path = "../../data/train/nevus"
nevus_valid_path = "../../data/valid/nevus"
nevus_test_path = "../../data/test/nevus"
seborrheic_train_path = "../../data/train/seborrheic_keratosis"
seborrheic_valid_path = "../../data/valid/seborrheic_keratosis"
seborrheic_test_path = "../../data/test/seborrheic_keratosis"

# Load images as numpy arrays. 
(X_train, X_valid, X_test,
 y_train, y_valid, y_test) = load_images(melanoma_train_path, 
                                         melanoma_valid_path,
                                         melanoma_test_path,
                                         nevus_train_path,
                                         nevus_valid_path,
                                         nevus_test_path,
                                         seborrheic_train_path, 
                                         seborrheic_valid_path,
                                         seborrheic_test_path)


(X_train_bottle, X_valid_bottle, X_test_bottle) = bottleneck_features(X_train,
                                                                      X_valid,
                                                                      X_test)

# Define fully connected layers.
model = Sequential()
model.add(Flatten(input_shape=(7, 7, 512)))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
              metrics=['accuracy'])
model.summary()

# Train the final layers.
checkpointer = ModelCheckpoint(filepath='cancervgg16.weights.best.hdf5',
                               verbose=1, save_best_only=True)

model.fit(X_train_bottle, y_train, epochs=20,
          validation_data=(X_valid_bottle, y_valid),
          callbacks=[checkpointer], verbose=1, shuffle=True)

model.load_weights('cancervgg16.weights.best.hdf5')

# Predict on test bottleneck features.
score, acc = model.evaluate(X_test_bottle, y_test, verbose = 1)
print('Test score: {:.2f}%'.format(score*100))
print('Test accuracy: {:.2f}%'.format(acc*100))

# Save predictions to csv for evaluation.
y_pred = model.predict(X_test_bottle)
task_1 = y_pred[:,0].reshape(-1,1)
task_2 = y_pred[:,2].reshape(-1,1)

evaluations = pd.read_csv('evaluation/sample_predictions.csv')
evaluations['task_1'] = task_1
evaluations['task_2'] = task_2
evaluations.to_csv('evaluation/evaluation.csv')
