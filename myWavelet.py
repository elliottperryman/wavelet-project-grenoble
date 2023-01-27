#!/usr/bin/env python
# coding: utf-8

# https://www.kaggle.com/datasets/saurabhshahane/barkvn50

import matplotlib.pyplot as plt
from keras.layers import (
    Input, Dense, Conv2D, Flatten, Reshape, Dropout, 
    Activation, AveragePooling2D, BatchNormalization,
    Lambda, Concatenate
)
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import image_dataset_from_directory

train = image_dataset_from_directory('./KTH-TIPS2-b', label_mode='categorical', seed=0, subset='training', validation_split=0.2)
val = image_dataset_from_directory('./KTH-TIPS2-b', label_mode='categorical', seed=0, subset='validation', validation_split=0.2)


checkpoint = ModelCheckpoint('./checkpoints/', save_best_only=True)


def WaveletTransform(img):
    low = img[:,::2,...] + img[:,1::2,...]
    low = low[:,:,::2,...] + low[:,:,1::2,...]
    high = img[:,::2,...] - img[:,1::2,...]
    high = high[:,:,::2,...] - high[:,:,1::2,...]
    return low, high


def conv_layer(_in, N):
    conv1 = Conv2D(N, kernel_size=(3,3), padding='same')(_in)
    norm1 = BatchNormalization()(conv1)
    relu1 = Activation('relu')(norm1)
    conv2 = Conv2D(N, kernel_size=(3,3), strides=(2,2), padding='same')(relu1)
    norm2 = BatchNormalization()(conv2)
    relu2 = Activation('relu')(norm2)
    return relu2


def build_model(input_shape = (256, 256, 3), num_classes=50):
    _input = BatchNormalization()(Input(input_shape))
    
    low1, high1 = Lambda(WaveletTransform, name='wavelet_1')(_input)
    low2, high2 = Lambda(WaveletTransform, name='wavelet_2')(low1)
    low3, high3 = Lambda(WaveletTransform, name='wavelet_3')(low2)
    low4, high4 = Lambda(WaveletTransform, name='wavelet_4')(low3)
    
    k1 = Concatenate()([conv_layer(_input,64), low1])
    k2 = Concatenate()([conv_layer(k1, 128), low2])
    k3 = Concatenate()([conv_layer(k2, 256), low3])
    k4 = Concatenate()([conv_layer(k3, 512), low4, high4])
    
    # k1 = conv_layer(_input,64)
    # k2 = conv_layer(k1, 128)
    # k3 = conv_layer(k2, 256)
    # k4 = conv_layer(k3, 512)
    
    avg_pool = AveragePooling2D(pool_size=(7,7), strides=1, padding='same')(k4)
    flat = Flatten()(avg_pool)
    output = Dense(num_classes, activation='softmax',name='fc')(flat)
    model = Model(inputs=_input, outputs=output)
    return model

n = next(train.as_numpy_iterator())
shape, num_classes = n[0].shape[1:], n[1].shape[-1]
model = build_model(input_shape=shape, num_classes=num_classes)

print(model.summary())

# plot_model(model)


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


hist = model.fit(train, validation_data=val, epochs=100, callbacks=[checkpoint])



