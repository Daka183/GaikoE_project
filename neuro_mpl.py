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

class MPL():
    """Построение, обучение и взаимодейтсвие с MPL моделью"""

    def __init__(self):
        self.word_count = -1            # Общее количество слов
        self.quantity_category = -1     # Количество категорий
        self.model = None
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
    
    # Загружаем данные для обучения из файлов
    def load_data(self):
        self.x_train = np.loadtxt('x_train_vec.txt', dtype=int)
        self.y_train = np.loadtxt('y_train_vec.txt', dtype=int)
        self.x_test = np.loadtxt('x_test_vec.txt', dtype=int)
        self.y_test = np.loadtxt('y_test_vec.txt', dtype=int)
    
    # Загружаем информацию о данных с которыми предстоит работа
    def load_data_info(self):
        data_info = np.loadtxt('data_info.txt', dtype=int)
        self.word_count = data_info[0]
        self.quantity_category = data_info[1]

    # Создание модели нейронной сети
    def create_model(self):
        self.model = Sequential()
        # Первый слой:
        self.model.add(Dense(512, input_shape=(self.word_count,)))  # Первое значене указывает на количество нейронов в слое 
        self.model.add(Activation('relu'))                          # Функция активации
        self.model.add(Dropout(0.2))                                # Борьба с переобучением

        # Второй слой:
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

        # Выходной слой:
        self.model.add(Dense(self.quantity_category))
        self.model.add(Activation('softmax'))

        # Настройка модели
        self.model.compile(loss=settings.loss_mpl,  # Функция потерь нейронной сети   
                optimizer=settings.optimizer_mpl,   # Оптимизатор нейронной сети
                metrics=["acc"])                    # Метрики для оценки нейронной сети

        # Вывод информации о слоях
        print(self.model.summary())          

    # Обучение модели
    def training_model(self):
        history = self.model.fit(self.x_train, self.y_train,
                    validation_split=settings.validation_split_mpl,
                    batch_size=settings.batch_size_mpl,
                    epochs=settings.epochs_mpl,
                    verbose=settings.verbose)
        
        # Сохраненеие истории обучения
        with open('trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        # Вывод информации о результатах обучения
        score = self.model.evaluate(self.x_test, self.y_test,
                       batch_size=settings.batch_size_mpl, verbose=settings.verbose)
        
        print()
        print(u'Оценка теста: {}'.format(score[0]))
        print(u'Оценка точности модели: {}'.format(score[1]))

    # Сохранение построеной модели
    def save_model(self):
        self.model.save('mpl_model')

    # Загрузка ранее построенной модели
    def load_model(self):
        self.model = load_model('mpl_model')

    # Загрузка словаря
    def load_token(self):
        with open('tokenizer.pickle', 'rb') as handle:
            dict = pickle.load(handle)
        tokenizer = Tokenizer(num_words=self.word_count)
        
        return dict, tokenizer

    # Выдача предсказания с помощью построенной модели
    def predict(self, dict, tokenizer, text):
        textSequences = dict.texts_to_sequences(text)                               # Замена слов на числа по частоте встречаемости
        text_vec = tokenizer.sequences_to_matrix(textSequences, mode='binary')      # Векторизация
        res = self.model.predict(text_vec)                                          # Выдача предсказания

        return np.argmax(res)

    # Создание графиков процесса обучения
    def graw_plot(self):
        with open('trainHistoryDict', 'rb') as file_pi:
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

