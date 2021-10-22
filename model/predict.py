#%%

import json
import os

import sys
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from lime import lime_image
from skimage.segmentation import mark_boundaries

from normalization import normalize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_PATH_COL = 'Image Path'
threshold = [0.12114254, 0.112214245, 0.01295178, 0.059730425, 0.07522124]

with open(BASE_DIR + '/constant.json', 'r') as const_json:
    const_dict = json.load(const_json)
    
    MODEL_PATH = const_dict['ModelPath']
    IMAGE_SIZE = const_dict['ImageSize']
    LABEL_LIST = const_dict['LabelList']

LABEL_LIST.remove('Mass')
LABEL_LIST.remove('Nodule')
LABEL_LIST.append('Mass / Nodule')

file_name = sys.argv[1]
result_dir = BASE_DIR + '/result/' + file_name

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

img = cv2.imread(BASE_DIR + '/../public/uploads/' + file_name + '.png')
img = normalize(img)
multi_disease_model = load_model(BASE_DIR + "/" + MODEL_PATH)

plt.imsave(result_dir + '/sample.png', img, cmap='gray')

saved = cv2.imread(result_dir + '/sample.png')

imageDataGenerator = ImageDataGenerator(samplewise_center=True,
                                        samplewise_std_normalization=True)

df = pd.DataFrame({IMAGE_PATH_COL: [result_dir + '/sample.png'], 'y': [0]})

gen = imageDataGenerator.flow_from_dataframe(dataframe=df,
                                             directory=None,
                                             x_col=IMAGE_PATH_COL,
                                             y_col=['y'],
                                             class_mode='raw',
                                             target_size=IMAGE_SIZE,
                                             color_mode='grayscale',
                                             batch_size=1)

x, y = next(gen)

target_img = x[0]

color_img = []

for i in range(0, IMAGE_SIZE[0]):
    color_img.append([])
    
    for j in range(0, IMAGE_SIZE[1]):
        color_img[i].append([target_img[i][j][0], target_img[i][j][0], target_img[i][j][0]])

color_img = np.array(color_img, dtype=np.double)

def grayscale_predict(image):
    image = np.delete(image, (1, 2), axis=3)
    
    return multi_disease_model.predict(image)


explainer = lime_image.LimeImageExplainer(random_state=7015)
explanation = explainer.explain_instance(color_img, grayscale_predict)

pred = multi_disease_model.predict(np.array([target_img]))

img = cv2.resize(img, dsize=IMAGE_SIZE)

pred_result = [0, 0, 0, 0, 0]

for idx, label in enumerate(LABEL_LIST):
    if pred[0][idx] >= threshold[idx]:
        image, mask = explanation.get_image_and_mask(
            idx,
            positive_only=True,            hide_rest=False,
            num_features=1
        )

        pred_result[idx] = 1
        
        if label == 'Mass / Nodule':
            plt.imsave(result_dir + '/Mass_or_Nodule.png', mark_boundaries(img, mask))
        else:
            plt.imsave(result_dir + '/' + label + '.png', mark_boundaries(img, mask))
            
print(pred_result)
sys.stdout.flush()
