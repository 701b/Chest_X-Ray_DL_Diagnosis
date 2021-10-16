import json

import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from lime import lime_image
from skimage.segmentation import mark_boundaries

from model.normalization import normalize

MODEL = 'DenseNet'
IMAGE_PATH_COL = 'Image Path'

with open('constant.json', 'r') as const_json:
    const_dict = json.load(const_json)
    
    MODEL_PATH = "xray_weights_resnet_(7).best.h5"
    IMAGE_SIZE = const_dict['ImageSize']
    LABEL_LIST = const_dict['LabelList']


print('loading image...')
img = cv2.imread('archive/images_001/images/00000013_013.png')
print('image loaded.')

print('resizing image...')
img = cv2.resize(img, dsize=IMAGE_SIZE)
print('image resized.')

print('normalizing image...')
img = normalize(img)
print('image normalized.')

print('loading model...')
multi_disease_model = load_model(MODEL_PATH)
print('model loaded.')

print(img)
plt.imshow(img)

explainer = lime_image.LimeImageExplainer(random_state=7015)
explanation = explainer.explain_instance(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), multi_disease_model.predict)

image, mask = explanation.get_image_and_mask(
    multi_disease_model.predict(
        np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)])
    ).argmax(axis=1)[0],
    positive_only=True,
    hide_rest=False
)

plt.imshow(mark_boundaries(image, mask))

pred = multi_disease_model.predict(np.array([img]))

for i in range(0, len(LABEL_LIST)):
    print(f"{LABEL_LIST[i]}: {pred[0][i]}")