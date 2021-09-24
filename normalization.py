import json

from skimage import exposure
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt


def normalization(path, filename):
    img = plt.imread(path)
    img = exposure.equalize_adapthist(img)
    plt.imsave('archive/normalized_images/' + filename, img, cmap='gray')


IMAGE_PATH_COL = 'Image Path'

with open('constant.json', 'r') as const_json:
    const_dict = json.load(const_json)
    
    MODEL_PATH = const_dict['ModelPath']
    IMAGE_SIZE = const_dict['ImageSize']
    LABEL_LIST = const_dict['LabelList']


xray_df = pd.read_csv('./archive/Data_Entry_2017.csv')

image_path_dict = {os.path.basename(x): x for x in glob(os.path.join('archive', 'images*', 'images', '*.png'))}
xray_df[IMAGE_PATH_COL] = xray_df['Image Index'].map(image_path_dict.get)

xray_df = xray_df.drop(['Finding Labels', 'Follow-up #', 'Patient ID', 'Patient Age', 'Patient Gender', 'View Position', 'OriginalImage[Width', 'Height]', 'OriginalImagePixelSpacing[x', 'y]', 'Unnamed: 11'],
                       axis='columns')

for index, row in xray_df.iterrows():
    if os.path.isfile('./archive/normalized_images/' + row['Image Index']):
        print(f'skip {row[IMAGE_PATH_COL]} [{index}]')
    else:
        normalization(row[IMAGE_PATH_COL], row['Image Index'])
        print(f'normalize {row[IMAGE_PATH_COL]} [{index}]')
