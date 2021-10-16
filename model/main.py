import numpy as np
import pandas as pd
import os
from glob import glob
from itertools import chain

from keras import Sequential
from keras.applications.mobilenet import MobileNet
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import load_model


def flow_from_df(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    df_gen = img_data_gen.flow_from_directory(base_dir, class_mode='sparse', **dflow_args)
    
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = ''
    
    return df_gen


# 최소 데이터 개수
MIN_COUNT = 1000

# 스케일링한 이미지 크기
IMG_SIZE = (128, 128)

# 저장된 가중치 경로
MODEL_PATH = 'xray_weights.densenet.h5'

# csv 파일로부터 xray 이미지 정보 가져오기
xray_df = pd.read_csv('archive/Data_Entry_2017.csv')

# df에 이미지 경로 포함시키기
image_path_dict = {os.path.basename(x): x for x in glob(os.path.join('archive', 'images*', 'images', '*.png'))}
xray_df['Image Path'] = xray_df['Image Index'].map(image_path_dict.get)

# df로부터 질환 라벨 추출
label_list = np.unique(list(chain(*xray_df['Finding Labels'].map(lambda x: x.split('|')))))
label_list = [x for x in label_list if x != 'No Finding']

# df에 각 라벨별로 열을 추가하고, 해당하는 열에 값을 1로 설정
for label in label_list:
    xray_df[label] = xray_df['Finding Labels'].map(lambda finding: 1.0 if label in finding else 0)

# 데이터가 일정 개수 이하인 라벨 제거
label_list = [x for x in label_list if xray_df[x].sum() > MIN_COUNT]

# 양은 많지만 의미가 적은 데이터 개수 줄이기 (질병 없음과 같은)
weights = xray_df['Finding Labels'].map(lambda x: len(x.split('|')) if x != 'No Finding' else 0).values + 0.1
weights /= weights.sum()
xray_df = xray_df.sample(40000, weights=weights)

# 질환이 있는지 없는지 구분하는 벡터 추가
xray_df['disease_vec'] = xray_df.apply(lambda x: [x[label_list].values], 1).map(lambda x: x[0])

# 학습용, 검증용, 테스트용으로 데이터 나누기
train_df, valid_df = train_test_split(xray_df, test_size=0.2, random_state=2021)

img_data_gen_for_train = ImageDataGenerator(samplewise_center=True,
                                            samplewise_std_normalization=True,
                                            horizontal_flip=True,
                                            vertical_flip=False,
                                            height_shift_range=0.05,
                                            width_shift_range=0.1,
                                            rotation_range=5,
                                            shear_range=0.1,
                                            fill_mode='reflect',
                                            zoom_range=0.15)

base_dir = os.path.dirname(train_df['Image Path'].values[0])

train_gen = img_data_gen_for_train.flow_from_dataframe(dataframe=train_df,
                                                       directory=None,
                                                       x_col='Image Path',
                                                       y_col=label_list,
                                                       class_mode='raw',
                                                       batch_size=64,
                                                       shuffle=True,
                                                       seed=1,
                                                       target_size=IMG_SIZE,
                                                       color_mode='grayscale')

raw_train_gen = ImageDataGenerator().flow_from_dataframe(dataframe=train_df,
                                                         directory=None,
                                                         x_col='Image Path',
                                                         y_col=label_list,
                                                         class_mode='raw',
                                                         batch_size=64,
                                                         shuffle=True,
                                                         target_size=IMG_SIZE,
                                                         color_mode='grayscale')

batch = raw_train_gen.next()
data_sample = batch[0]

img_data_gen_for_valid = ImageDataGenerator(featurewise_center=True,
                                            featurewise_std_normalization=True)

img_data_gen_for_valid.fit(data_sample)

valid_gen = img_data_gen_for_valid.flow_from_dataframe(dataframe=valid_df,
                                                       directory=None,
                                                       x_col='Image Path',
                                                       y_col=label_list,
                                                       class_mode='raw',
                                                       batch_size=64,
                                                       shuffle=False,
                                                       seed=1,
                                                       target_size=IMG_SIZE,
                                                       color_mode='grayscale')

# test_gen = img_data_gen_for_valid.flow_from_dataframe(dataframe=test_df,
#                                                       directory=None,
#                                                       x_col='Image Path',
#                                                       y_col='disease_vec',
#                                                       class_mode='raw',
#                                                       batch_size=64,
#                                                       shuffle=False,
#                                                       seed=1,
#                                                       target_size=IMG_SIZE,
#                                                       color_mode='grayscale')

base_mobilenet_model = MobileNet(input_shape=(*IMG_SIZE, 1),
                                 include_top=False,
                                 weights=None)
multi_disease_model = Sequential()
multi_disease_model.add(base_mobilenet_model)
multi_disease_model.add(GlobalAveragePooling2D())
multi_disease_model.add(Dropout(0.5))
multi_disease_model.add(Dense(512))
multi_disease_model.add(Dropout(0.5))
multi_disease_model.add(Dense(len(label_list), activation='sigmoid'))
multi_disease_model.compile(optimizer='adam',
                            loss='binary_crossentropy',
                            metrics=['binary_accuracy', 'mae'])
multi_disease_model.summary()

checkpoint = ModelCheckpoint(MODEL_PATH,
                             monitor='val_binary_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode='max')

early = EarlyStopping(monitor='val_binary_accuracy',
                      mode='max',
                      patience=3)

callback_list = [checkpoint, early]

multi_disease_model.fit(train_gen,
                        steps_per_epoch=100,
                        validation_data=valid_gen,
                        validation_steps=100,
                        epochs=5,
                        callbacks=callback_list)

multi_disease_model = load_model(MODEL_PATH)

# pred = multi_disease_model.predict(test_gen, steps=len(test_gen))
