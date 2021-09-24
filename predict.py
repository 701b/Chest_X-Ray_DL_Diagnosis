import cv2
import numpy as np

from keras.models import load_model

MODEL_PATH = 'xray_weights.densenet.h5'
IMG_SIZE = (128, 128)

LABEL_LIST = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

print('loading image...')
img = cv2.imread('./archive/images_001/images/00000013_013.png')
print('image loaded.')

print('resizing image...')
img = cv2.resize(img, dsize=IMG_SIZE)
print('image resized.')

print('loading model...')
multi_disease_model = load_model(MODEL_PATH)
print('model loaded.')

pred = multi_disease_model.predict(np.array([img / 256]))

for i in range(0, len(LABEL_LIST)):
    print(f"{LABEL_LIST[i]}: {pred[0][i]}")