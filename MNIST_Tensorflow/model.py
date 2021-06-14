# Library imports
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

'''
Configuration for model params
'''
INPUT_SHAPE = (28, 28)
MODEL_PATH = 'my_model'


# load model
model = load_model(MODEL_PATH)


def predict(img):
    """
    Return single class
    """
    img_arr = preprocessing(img)
    prediction = model.predict(img_arr)
    predict_class = np.argmax(prediction)
    return predict_class


# pre-processs image
def preprocessing(img):
    """
    load image data, resize into input shape required by model; normalise; add dim
    :return:
    """
    img = Image.open(img).convert('L')
    img = img.resize(INPUT_SHAPE)
    img_arr = np.asarray(img) / 255
    img_arr = np.expand_dims(img_arr[:, :, np.newaxis], 0)
    return img_arr

