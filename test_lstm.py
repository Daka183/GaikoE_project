from asyncio.windows_events import NULL
from unittest import result
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Embedding, LSTM

import numpy as np
import pickle
import matplotlib.pyplot as plt
import settings

x_train = np.loadtxt('x_train_vec_LSTM.txt', dtype=int)
y_train = np.loadtxt('y_train_vec_LSTM.txt', dtype=int)
x_test = np.loadtxt('x_test_vec_LSTM.txt', dtype=int)
y_test = np.loadtxt('y_test_vec_LSTM.txt', dtype=int)

data_info = np.loadtxt('data_info_LSTM.txt', dtype=int)
word_count = data_info[0]
quantity_category = data_info[1]
max_word = data_info[3]


optim = ['Adam', 'SGD', 'RMSprop']
file = open("test_lstm.txt", 'w', encoding='utf-8')
model = None

def optAndLayer(neuro, e, m_layer):
    layer = 1
    epochs_lstm = e

    o = 0
    while layer != m_layer + 1:
        
        model = Sequential()
        model.add(Embedding(word_count+1, max_word))
        l = 1
        opt = optim[o] 
        
        while l < layer + 1:
            if l + 1 == layer + 1:
                model.add(LSTM(round(neuro/l), dropout=0.2, recurrent_dropout=0.2))
            else:
                model.add(LSTM(round(neuro/l), dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
            
            l = l + 1                              

        model.add(Dense(quantity_category, activation='sigmoid'))

        model.compile(loss=settings.loss_lstm,optimizer=opt, metrics=['acc']) 
        history = model.fit(x_train, y_train, batch_size=settings.batch_size_lstm, epochs=epochs_lstm, validation_data=(x_test, y_test))                    
        score = model.evaluate(x_test, y_test, batch_size=settings.batch_size_lstm, verbose=settings.verbose)
        
        file.write(u'Оптимизатор: {}. '.format(opt))
        file.write(u'Количество слоёв: {}. '.format(l))
        file.write(u'Количеств нейронов: {}. '.format(neuro))
        file.write(u'Количество эпох:: {}. '.format(epochs_lstm))
        file.write(u'Оценка теста: {}. '.format(score[0]))
        file.write(u'Оценка точности модели: {}.\n\n'.format(score[1]))

        print()
        print(u'Оценка теста: {}'.format(score[0]))
        print(u'Оценка точности модели: {}'.format(score[1]))
        
        model = None
        o = o + 1
        if o == 3:
            layer = layer + 1
            o = 0

n = 256
max_layer = 4
while n != 4096:
    e = 5
    while e != 15:
        optAndLayer(n, e, max_layer)
        e = e + 1
    
    n = n * 2