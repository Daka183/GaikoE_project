import pandas
import pickle
import numpy as np
import settings
from tensorflow import keras
from keras.preprocessing.text import Tokenizer

class Tokenization():
    """Используется Keras для подготовки данных"""

    def __init__(self, path_file, sheet_number):
        self.path_file = path_file              # Путь к файлу с данными для обработки
        self.sheet_number = sheet_number        # Номер странцы
        # Дата фрэйма с данными для обработи
        self.exel_data_df = pandas.read_excel(path_file, sheet_name=sheet_number)
        self.size_exel_data_df = len(self.exel_data_df) # Размер дата фрейма
        #self.size_exel_data_df = 50
        self.quantity_category = -1     # Количество категорий
        self.tokenizer = Tokenizer()    # Инициализация токенизатора
        self.textSequences = []         # Массив слов закодированых в числах
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.total_words = -1           # Количество всех найденых слов
        self.word_count = -1            # Количество слов для обработки

    # Расчёт количества категорий
    def quantity_categories(self, path_file, sheet_number):
        exel_category_df = pandas.read_excel(path_file, sheet_name=sheet_number)
        self.quantity_category = len(exel_category_df)
    
    # Формирование словая
    def creation_dictionary(self, model):
        if model == 1:
            path_token = 'tokenizer_mpl.pickle'
        if model == 2:
            path_token = 'tokenizer_lstm.pickle'
        
        # Собираем все текстовые запросы в массив
        i = 0
        text_data = []
        while i < self.size_exel_data_df:
            text_data.append(str(self.exel_data_df.values[i][1]))
            i = i + 1 

        # Создание словаря (слово -> число)
        self.tokenizer.fit_on_texts(text_data)
        # Расчёт количетва слов в словаре
        self.total_words = len(self.tokenizer.word_index)
        # Замена слов на числа по словарю
        self.textSequences = self.tokenizer.texts_to_sequences(text_data)
        print(self.textSequences[0])

        print('В словаре {} слов'.format(self.total_words))
        print('Укажите число слов которые будут использованы в векторизации:')
        self.word_count = int(input())

        # Создаём словарь с указаным количеством слов
        self.tokenizer = Tokenizer(num_words=self.word_count)
        # Создание словаря (слово -> число)
        self.tokenizer.fit_on_texts(text_data)
        # Расчёт количетва слов в словаре
        self.total_words = len(self.tokenizer.word_index)
        # Замена слов на числа по словарю
        self.textSequences = self.tokenizer.texts_to_sequences(text_data)
        print(self.textSequences[0])

        # Сохранение токенизатора
        with open(path_token, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Токенизированным данным раскладываем по масивам для тренировки и тестов
    def data_from_arrays(self, train_test_split=0.9):
        i = 0
        categories = []
        while i < self.size_exel_data_df:
            categories.append((self.exel_data_df.values[i][2]))
            i = i + 1
    
        test_size = int(self.size_exel_data_df - round(self.size_exel_data_df * train_test_split))
        print("Test size: {}".format(test_size))
    
        print("\nTraining set:")
        self.x_train = self.textSequences[test_size:]
        print("\t - x_train: {}".format(len(self.x_train)))
        self.y_train = categories[test_size:]
        print("\t - y_train: {}".format(len(self.y_train)))
    
        print("\nTesting set:")
        self.x_test = self.textSequences[:test_size]
        print("\t - x_test: {}".format(len(self.x_test)))
        self.y_test = categories[:test_size]
        print("\t - y_test: {}".format(len(self.y_test)))

    # Векторизация полученных массивов для MPL
    def vectorization_arrays(self):
        print(u'Преобразуем описания заявок в векторы чисел')
        
        # Векторизация текстовых данных
        x_train_vec = self.tokenizer.sequences_to_matrix(self.x_train, mode='binary')
        x_test_vec = self.tokenizer.sequences_to_matrix(self.x_test, mode='binary')
        print('Размерность X_train:', x_train_vec.shape)
        print('Размерность X_test:', x_test_vec.shape)

        # Векторизация категорий
        y_train_vec = keras.utils.to_categorical(self.y_train, self.quantity_category)
        y_test_vec = keras.utils.to_categorical(self.y_test, self.quantity_category)
        print('y_train shape:', y_train_vec.shape)
        print('y_test shape:', y_test_vec.shape)

        # Сохраняем инормацию о сформированных данных
        stat = np.array([self.word_count, self.quantity_category, self.total_words])
        np.savetxt('x_train_vec.txt', x_train_vec, fmt='%d')
        np.savetxt('x_test_vec.txt', x_test_vec, fmt='%d')
        np.savetxt('y_train_vec.txt', y_train_vec, fmt='%d')
        np.savetxt('y_test_vec.txt', y_test_vec, fmt='%d')
        np.savetxt('data_info.txt', stat, fmt='%d')

    # Векторизация полученных массивов для LSTM
    def vectorization_arrays_LSTM(self, max_word):
        print(u'Преобразуем описания заявок в векторы чисел')

        # Векторизация текстовых данных
        x_train_vec = keras.preprocessing.sequence.pad_sequences(self.x_train, maxlen=max_word)
        x_test_vec = keras.preprocessing.sequence.pad_sequences(self.x_test, maxlen=max_word)
        print('Размерность X_train:', x_train_vec.shape)
        print('Размерность X_test:', x_test_vec.shape)

        # Векторизация категорий
        y_train_vec = keras.utils.to_categorical(self.y_train, self.quantity_category)
        y_test_vec = keras.utils.to_categorical(self.y_test, self.quantity_category)
        print('y_train shape:', y_train_vec.shape)
        print('y_test shape:', y_test_vec.shape)

        # Сохраняем инормацию о сформированных данных
        stat = np.array([self.word_count, self.quantity_category, self.total_words, max_word])
        np.savetxt('x_train_vec_LSTM.txt', x_train_vec, fmt='%d')
        np.savetxt('x_test_vec_LSTM.txt', x_test_vec, fmt='%d')
        np.savetxt('y_train_vec_LSTM.txt', y_train_vec, fmt='%d')
        np.savetxt('y_test_vec_LSTM.txt', y_test_vec, fmt='%d')
        np.savetxt('data_info_LSTM.txt', stat, fmt='%d')

        
