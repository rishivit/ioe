import tensorflow as tf
# import matplotlib.pyplot as plt

import numpy as np
import platform
import pathlib
import os
# import PIL
# from PIL import Image
import os
import PIL
import PIL.Image


# GLOBALS DECLARATION
INPUT_IMAGE_SIZE = 224
TEST_IMAGES_DIR_PATH = pathlib.Path('data')
TEST_IMAGE_PATHS = sorted(list(TEST_IMAGES_DIR_PATH.glob('*.jpg')))
TEST_IMAGE_PATHS
TEST_IMAGE_INDEX = 1

# Loading Images
def load_image(image_path):
    return tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=[INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE]
    )

def image_to_array(image):
    return tf.keras.preprocessing.image.img_to_array(image, dtype=np.int32)


# Preprocess Images
def image_preprocess(image_array):
    return tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array[tf.newaxis, ...]
    )

# Classify image
def get_tags(probs, labels, max_classes = 5, prob_threshold = 0.01):
    probs_mask = probs > prob_threshold
    probs_filtered = probs[probs_mask] * 100
    labels_filtered = labels[probs_mask]
    
    sorted_index = np.flip(np.argsort(probs_filtered))
    labels_filtered = labels_filtered[sorted_index][:max_classes]
    probs_filtered = probs_filtered[sorted_index][:max_classes].astype(np.int)
    
    tags = ''
    for i in range(0, len(labels_filtered)):
        tags = tags + labels_filtered[i] + ' (' + str(probs_filtered[i]) + '%), ' 

    return tags, labels_filtered, probs_filtered 

def getClassifications():
    # Load model
    model  = tf.keras.models.load_model(os.path.join("./model/","image_classification_mobilenet_v2.h5"))
    # print(model.summary())

    # Load labels
    labels_path = os.path.join('./', 'labels.txt')
    labels = np.array(
        open(labels_path).read().splitlines()
    )[1:]

    # Load images

    test_images = []
    for image_path in TEST_IMAGE_PATHS:
        # <PIL.Image.Image image mode=RGB size=224x224 at 0x141247ED0>
        test_image = load_image(image_path)
        test_image_array = image_to_array(test_image)
        test_images.append(test_image_array)

    # Preprocessing
    test_images_preprocessed = []
    for test_image in test_images:
        test_image_preprocessed = image_preprocess(test_image)
        test_images_preprocessed.append(test_image_preprocessed)

    # print('Image shape before preprocessing:', test_images[0].shape)
    # print('Image shape after preprocessing:', test_images_preprocessed[0].shape)


    result = model(test_images_preprocessed[TEST_IMAGE_INDEX])
    np_result = result.numpy()[0]

    tags, labels_filtered, probs_filtered = get_tags(np_result, labels)
    
    # print('probs_filtered:', probs_filtered)
    # print('labels_filtered:', labels_filtered)
    # print('tags:', tags)

    return dict([
        ('probs_filtered', probs_filtered),
        ('labels_filtered', labels_filtered),
        ('tags', tags)
    ])

# Driver function
def main():
    if __name__ == '__main__':
        results = getClassifications()
        print(results['labels_filtered'])

main()