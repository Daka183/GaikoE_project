from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.models import load_model
from keras.preprocessing.text import Tokenizer

import numpy as np
import pickle
import matplotlib.pyplot as plt
import settings

class LSTM_model():
    
    def __init__(self):
        self.word_count = -1
        self.quantity_category = -1
        self.max_word = -1
        self.model = None
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

    def load_data(self):
        self.x_train = np.loadtxt('x_train_vec_LSTM.txt', dtype=int)
        self.y_train = np.loadtxt('y_train_vec_LSTM.txt', dtype=int)
        self.x_test = np.loadtxt('x_test_vec_LSTM.txt', dtype=int)
        self.y_test = np.loadtxt('y_test_vec_LSTM.txt', dtype=int)

    def load_data_info(self):
        data_info = np.loadtxt('data_info_LSTM.txt', dtype=int)
        self.word_count = data_info[0]
        self.quantity_category = data_info[1]
        self.max_word = data_info[3]

    def create_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.word_count+1, self.max_word))
        self.model.add(LSTM(32, input_shape=(self.max_word,), dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(self.quantity_category, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

        print(self.model.summary())

    def training_model(self):
        history = self.model.fit(self.x_train, self.y_train,
                    batch_size=32,
                    epochs=5,
                    validation_data=(self.x_test, self.y_test))

        with open('trainHistoryDictLstm', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        score = self.model.evaluate(self.x_test, self.y_test,
                       batch_size=32, verbose=1)
        print()
        print(u'Оценка теста: {}'.format(score[0]))
        print(u'Оценка точности модели: {}'.format(score[1]))

    def save_model(self):
        self.model.save('lstm_model')

    def load_model(self):
        self.model = load_model('lstm_model')

    def load_token(self):
        with open('tokenizer.pickle', 'rb') as handle:
            dict = pickle.load(handle)
        tokenizer = Tokenizer(num_words=self.word_count)
        
        return dict, tokenizer

    def save_model(self):
        self.model.save('lstm_model')

    def load_model(self):
        self.model = load_model('lstm_model')

    def load_token(self):
        with open('tokenizer.pickle', 'rb') as handle:
            dict = pickle.load(handle)
        tokenizer = Tokenizer(num_words=self.word_count)
        
        return dict, tokenizer
    
    def predict(self, dict, tokenizer, text):
        textSequences = dict.texts_to_sequences(text)
        text_vec= keras.preprocessing.sequence.pad_sequences(textSequences, maxlen=self.max_word)
        res = self.model.predict(text_vec)

        return np.argmax(res)

    def graw_plot(self):
        with open('trainHistoryDictLstm', 'rb') as file_pi:
            history = pickle.load(file_pi)

        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # График оценки loss
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
