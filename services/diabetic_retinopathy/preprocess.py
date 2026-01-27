import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMG_SIZE = (150, 150)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img = img_to_array(img)
    img = img / 255.0           # same normalization as training
    img = np.expand_dims(img, axis=0)
    return img
