# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 14:41:40 2021

@author: kevin
"""

import requests
from io import StringIO
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import datetime
import calendar
from sklearn.preprocessing import normalize
import time
import math
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf 
from keras.layers import GaussianNoise , BatchNormalization
from sklearn.cluster import KMeans



def get_series_data( df, period, isNormaliztion) -> list:
    list_ = []
    for i in range(len(df)-period+1):
        if isNormaliztion :
            list_.append(normalize((((df[i:i+period].values)/df.Close[i])-1)*1000, axis=0))
        else :
            list_.append((((df[i:i+period].values)/df.Close[i])-1)*1000)
        
    return np.array(list_)


def triple_barrier(price, ub, lb, period):

    ## 取超出邊界的資料，記錄第一筆為邊界收益，沒超出的話取時間內最後一筆
    def record_price(s):
        return np.append(s[(s / s[0] > ub) | (s / s[0] < lb)], s[-1])[0]/s[0]
    ## 今天往後20天，所以+1
    p = price.rolling(period+1).apply(record_price, raw=True).shift(-period)

    signal = pd.Series(0, p.index)
    signal.loc[p > ub] = 1
    signal.loc[p < lb] = -1
    ret = pd.DataFrame({'triple_barrier_profit':p, 'triple_barrier_signal':signal})

    return ret

def cnn_training(allData, allLabel, day , splitsize, epoches = 100) :

    week_list = allData
    week_label = allLabel
    # 定義梯度下降批量
    batch_size = 32
    # 定義分類數量
    num_classes = 3
    # 定義訓練週期
    epochs = epoches

    # 定義圖像寬、高
    img_rows, img_cols = day, 4
    input_shape = ( img_rows, img_cols)

    # 載入 MNIST 訓練資料
    split_ratio = splitsize
    x_train = week_list[ math.ceil(len(week_list)*split_ratio) :]
    x_test = week_list[ : math.ceil(len(week_list)*split_ratio) ]

    y_train = week_label[ math.ceil(len(week_label)*split_ratio) :]
    y_test = week_label[ : math.ceil(len(week_label)*split_ratio) ]

    x_train = x_train.reshape(x_train.shape[0] , img_rows, img_cols,1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols ,1)

    # x_train  = torch.from_numpy(x_train)
    # x_test  = torch.from_numpy(x_test)

    # y_train = torch.from_numpy(y_train)
    # y_test = torch.from_numpy(y_test)

    input_shape = (img_rows, img_cols,1 )

    # 保留原始資料，供 cross tab function 使用
    y_test_org = y_test


    # y 值轉成 one-hot encoding
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    # 建立簡單的線性執行的模型
    model = Sequential()
    # 建立卷積層，filter=32,即 output space 的深度, Kernal Size: 3x3, activation function 採用 relu
    model.add(Conv2D(16, kernel_size=(3,2),
                    activation='relu',
                    input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.3))
    # 建立卷積層，filter=64,即 output size, Kernal Size: 3x3, activation function 採用 relu
    model.add(Conv2D(32, (3,2), activation='relu'))
    # 建立池化層，池化大小=2x2，取最大值
    model.add(MaxPooling2D(pool_size=(3, 2)))
    # Dropout層隨機斷開輸入神經元，用於防止過度擬合，斷開比例:0.25
    model.add(Dropout(0.25))
    # Flatten層把多維的輸入一維化，常用在從卷積層到全連接層的過渡。
    model.add(Flatten( name ='flatten'))
    # 全連接層: 128個output
    model.add(Dense(batch_size, 'sigmoid', name ='Dense'))
    # 使用 softmax activation function，將結果分類
    model.add(Dense(num_classes, activation='softmax' ))

    # 編譯: 選擇損失函數、優化方法及成效衡量方式
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer='adam',
                metrics=['accuracy'])

    # 進行訓練, 訓練過程會存在 train_history 變數中
    train_history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))

    # 顯示損失函數、訓練成果(分數)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score)
    print('Test accuracy:', score)
    return model

def get_cluster (allData, n_clusters) :
    X = np.array(allData)
    X = X.reshape(X.shape[0] , X.shape[2])

    kmeans_plus_plus = KMeans(n_clusters=n_clusters,
                    n_init=3,
                    init='k-means++',
                    tol=1e-4,
                    random_state=111,
                    verbose=False).fit(X)
    return kmeans_plus_plus.labels_