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
    """Построение, обучение и взаимодейтсвие с LSTM моделью"""

    # Инициализация класса
    def __init__(self):
        self.word_count = -1            # Общее количество слов
        self.quantity_category = -1     # Количество категорий
        self.max_word = -1              # Максимальное количество слов в строке
        self.model = None
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

    # Загружаем данные для обучения из файлов
    def load_data(self):
        self.x_train = np.loadtxt('x_train_vec_LSTM.txt', dtype=int)
        self.y_train = np.loadtxt('y_train_vec_LSTM.txt', dtype=int)
        self.x_test = np.loadtxt('x_test_vec_LSTM.txt', dtype=int)
        self.y_test = np.loadtxt('y_test_vec_LSTM.txt', dtype=int)

    # Загружаем информацию о данных с которыми предстоит работа
    def load_data_info(self):
        data_info = np.loadtxt('data_info_LSTM.txt', dtype=int)
        self.word_count = data_info[0]
        self.quantity_category = data_info[1]
        self.max_word = data_info[3]

    # Создание модели нейронной сети
    def create_model(self):
        self.model = Sequential()

        # Преобразование чисел в векторы
        self.model.add(Embedding(self.word_count+1, self.max_word))

        # LSTM слой, первое число количество нейронов
        self.model.add(LSTM(32, input_shape=(self.max_word,), dropout=0.2, recurrent_dropout=0.2))

        # Выходной слой
        self.model.add(Dense(self.quantity_category, activation='sigmoid'))

        # Настройка модели
        self.model.compile(loss=settings.loss_lstm,  # Функция потерь нейронной сети
              optimizer=settings.optimizer_lstm,     # Оптимизатор нейронной сети
              metrics=['acc'])                       # Метрики для оценки нейронной сети

        # Вывод информации о слоях
        print(self.model.summary())

    # Обучение модели
    def training_model(self):         
        history = self.model.fit(self.x_train, self.y_train,
                    batch_size=settings.batch_size_lstm,
                    epochs=settings.epochs_lstm,
        # Данные, по которым оцениваются потери и любые показатели модели в конце каждой эпохи.
                    validation_data=(self.x_test, self.y_test)) 

        # Сохраненеие истории обучения
        with open('trainHistoryDictLstm', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        # Вывод информации о результатах обучения
        score = self.model.evaluate(self.x_test, self.y_test,
                       batch_size=settings.batch_size_lstm, verbose=settings.verbose)
        print()
        print(u'Оценка теста: {}'.format(score[0]))
        print(u'Оценка точности модели: {}'.format(score[1]))

    # Сохранение построеной модели
    def save_model(self):
        self.model.save('lstm_model')

    # Загрузка ранее построенной модели
    def load_model(self):
        self.model = load_model('lstm_model')

    # Загрузка словаря
    def load_token(self):
        with open('tokenizer.pickle', 'rb') as handle:
            dict = pickle.load(handle)
        tokenizer = Tokenizer(num_words=self.word_count)
        
        return dict, tokenizer

    # Выдача предсказания с помощью построенной модели
    def predict(self, dict, tokenizer, text):
        textSequences = dict.texts_to_sequences(text)   # Замена слов на числа по частоте встречаемости
        text_vec= keras.preprocessing.sequence.pad_sequences(textSequences, maxlen=self.max_word)   # Векторизация
        res = self.model.predict(text_vec)              # Выдача предсказания

        return np.argmax(res)

    # Создание графиков процесса обучения
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
