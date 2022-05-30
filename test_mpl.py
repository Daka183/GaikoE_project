from asyncio.windows_events import NULL
from unittest import result
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.models import load_model
from keras.preprocessing.text import Tokenizer

import numpy as np
import pickle
import matplotlib.pyplot as plt
import settings

x_train = np.loadtxt('x_train_vec.txt', dtype=int)
y_train = np.loadtxt('y_train_vec.txt', dtype=int)
x_test = np.loadtxt('x_test_vec.txt', dtype=int)
y_test = np.loadtxt('y_test_vec.txt', dtype=int)

data_info = np.loadtxt('data_info.txt', dtype=int)
word_count = data_info[0]
quantity_category = data_info[1]

optim = ['Adam', 'SGD', 'RMSprop']
result_data = []

file = open("test_mpl.txt", 'w', encoding='utf-8')

model = None

max_layer = 1

def optAndLayer(neuro, e, m_layer):
    layer = 1
    epochs_mpl = e

    o = 0
    while layer != m_layer + 1:
        
        model = Sequential()
        l = 1
        opt = optim[o] 
        while l < layer + 1:
            model.add(Dense(neuro/l, input_shape=(word_count,)))        
            model.add(Activation('relu'))                          
            model.add(Dropout(0.2))
            l = l + 1                              

        model.add(Dense(quantity_category))
        model.add(Activation('softmax'))

        model.compile(loss=settings.loss_mpl, optimizer=opt, metrics=["acc"])                    
        history = model.fit(x_train, y_train, validation_split=0.25, batch_size=32, epochs=epochs_mpl, verbose=1)
        score = model.evaluate(x_test, y_test, batch_size=32, verbose=1)    

        file.write(u'Оптимизатор: {}. '.format(opt))
        file.write(u'Количество слоёв: {}. '.format(l))
        file.write(u'Количеств нейронов: {}. '.format(neuro))
        file.write(u'Количество эпох:: {}. '.format(epochs_mpl))
        file.write(u'Оценка теста: {}. '.format(score[0]))
        file.write(u'Оценка точности модели: {}.\n\n'.format(score[1]))
        
        model = None
        o = o + 1
        if o == 3:
            layer = layer + 1
            o = 0

n = 256
max_layer = 4
while n != 4096:
    e = 10
    while e != 20:
        optAndLayer(n, e, max_layer)
        e = e + 5
    
    n = n * 2




