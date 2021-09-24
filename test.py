import json
import os
from glob import glob

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from keras.applications.densenet import DenseNet121
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet import ResNet50
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.metrics import AUC
from keras.models import Sequential, load_model

from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

MODEL = 'DenseNet'
IMAGE_PATH_COL = 'Image Path'

with open('constant.json', 'r') as const_json:
    const_dict = json.load(const_json)
    
    MODEL_PATH = const_dict['ModelPath']
    IMAGE_SIZE = const_dict['ImageSize']
    LABEL_LIST = const_dict['LabelList']

# DataFrame 구성하기
xray_df = pd.read_csv('./archive/Data_Entry_2017.csv')

image_path_dict = {os.path.basename(x): x for x in glob(os.path.join('archive', 'filtered_normalized_images', '*.png'))}
xray_df[IMAGE_PATH_COL] = xray_df['Image Index'].map(image_path_dict.get)

xray_df = xray_df.drop(
    ['Follow-up #', 'Patient ID', 'Patient Age', 'Patient Gender', 'View Position', 'OriginalImage[Width', 'Height]',
     'OriginalImagePixelSpacing[x', 'y]', 'Unnamed: 11'], axis='columns')

xray_df.drop(xray_df.loc[xray_df[IMAGE_PATH_COL].isnull()].index, inplace=True)

for label in LABEL_LIST:
    xray_df[label] = xray_df['Finding Labels'].map(lambda finding: 1.0 if label in finding else 0)

xray_df.loc[((xray_df['Mass'] == 1) | (xray_df['Nodule'] == 1)), 'Mass / Nodule'] = 1
xray_df.loc[((xray_df['Mass'] != 1) & (xray_df['Nodule'] != 1)), 'Mass / Nodule'] = 0

xray_df = xray_df.drop(['Mass', 'Nodule'], axis='columns')

LABEL_LIST.remove('Mass')
LABEL_LIST.remove('Nodule')
LABEL_LIST.append('Mass / Nodule')

# No Finding 라벨 줄이기
# sample_weights = xray_df['Finding Labels'].map(lambda x: len(x.split('|')) if x != 'No Finding' else 0).values + 0.05
# sample_weights /= sample_weights.sum()
# xray_df = xray_df.sample(65000, weights=sample_weights)

label_counts = xray_df['Finding Labels'].value_counts()[:15]
fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
ax1.bar(np.arange(len(label_counts)) + 0.5, label_counts)
ax1.set_xticks(np.arange(len(label_counts)) + 0.5)
_ = ax1.set_xticklabels(label_counts.index, rotation=90)
plt.show()

# 데이터 나누기
train_df, valid_and_test_df = train_test_split(xray_df, test_size=0.3, random_state=7015)
valid_df, test_df = train_test_split(valid_and_test_df, test_size=0.5, random_state=7015)

# 데이터 Augmentation
image_data_generator = ImageDataGenerator(samplewise_center=True,
                                          samplewise_std_normalization=True,
                                          horizontal_flip=True,
                                          vertical_flip=False,
                                          rotation_range=5,
                                          shear_range=0.1,
                                          fill_mode='constant',
                                          cval=0,
                                          zoom_range=0.15,
                                          brightness_range=(0.8, 1.2))

image_data_generator_for_valid = ImageDataGenerator(samplewise_center=True,
                                                    samplewise_std_normalization=True)

train_gen = image_data_generator.flow_from_dataframe(dataframe=train_df,
                                                     directory=None,
                                                     x_col=IMAGE_PATH_COL,
                                                     y_col=LABEL_LIST,
                                                     class_mode='raw',
                                                     target_size=IMAGE_SIZE,
                                                     color_mode='grayscale',
                                                     batch_size=16)

valid_gen = image_data_generator_for_valid.flow_from_dataframe(dataframe=valid_df,
                                                               directory=None,
                                                               x_col=IMAGE_PATH_COL,
                                                               y_col=LABEL_LIST,
                                                               class_mode='raw',
                                                               target_size=IMAGE_SIZE,
                                                               color_mode='grayscale',
                                                               batch_size=128)

test_gen = image_data_generator_for_valid.flow_from_dataframe(dataframe=test_df,
                                                              directory=None,
                                                              x_col=IMAGE_PATH_COL,
                                                              y_col=LABEL_LIST,
                                                              class_mode='raw',
                                                              target_size=IMAGE_SIZE,
                                                              color_mode='grayscale',
                                                              batch_size=len(test_df))

test_X, test_Y = next(test_gen)

# 모델 불러오기
multi_disease_model = load_model(MODEL_PATH)


#
pred_Y = multi_disease_model.predict(test_X, batch_size=32, verbose=True)

fig, c_ax = plt.subplots(1, 1, figsize=(9, 9))
for (idx, c_label) in enumerate(LABEL_LIST):
    fpr, tpr, thresholds = roc_curve(test_Y[:, idx].astype(int), pred_Y[:, idx])
    c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('trained_net.png')
