# dermatologist-ai
Skin cancer detection project MLND from [Udacity repo](https://github.com/udacity/dermatologist-ai):
"The data and objective are pulled from the [2017 ISIC Challenge on Skin Lesion Analysis Towards Melanoma Detection](https://challenge.kitware.com/#challenge/583f126bcad3a51cc66c8d9a).  As part of the challenge, participants were tasked to design an algorithm to diagnose skin lesion images as one of three different skin diseases (melanoma, nevus, or seborrheic keratosis).  In this project, you will create a model to generate your own predictions."

Uses Keras VGG-16 trained on ImageNet data with final fully connected layers trained on skin cancer cell images.

Scored

Category 1 score (melanoma detection - area under ROC curve): 0.727 (17th place in ISIC challenge)

Category 2 score (nevus detection - area under ROC curve): 0.822 (18th place in ISIC challenge)

Category 3 score (seborrheic keratosis detection - area under ROC curve): 0.775 (18th place in ISIC challenge)

Threshold set to 0.05 for confusion matrix.

## Getting Started

1. Clone the [repository](https://github.com/cchadha2/dermatologist-ai) and create a `data/` folder to hold the dataset of skin images.  
```text
git clone https://github.com/cchadha2/dermatologist-ai.git
mkdir data; cd data
```
2. Create folders to hold the training, validation, and test images.
```text
mkdir train; mkdir valid; mkdir test
```
3. Download and unzip the [training data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip) (5.3 GB).

4. Download and unzip the [validation data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip) (824.5 MB).

5. Download and unzip the [test data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip) (5.1 GB).

6. Place the training, validation, and test images in the `data/` folder, at `data/train/`, `data/valid/`, and `data/test/`, respectively.  Each folder should contain three sub-folders (`melanoma/`, `nevus/`, `seborrheic_keratosis/`), each containing representative images from one of the three image classes.
