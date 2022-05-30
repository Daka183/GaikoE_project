import settings
import preprocessingData
import tokenizer
import neuro_mpl
import neuro_lstm

flag = -1

while flag != 0:
    print("------------------------------------------------")
    print("Выберите пункт из предложенных:")
    print("Для предварительной обработки данных введите ------- 1")
    print("Для подготовки данных к обучению введите ----------- 2")
    print("Для создание и обучения нейронной сети MPL введие -- 3")
    print("Для использвания нейронной сети MPL введите -------- 4")
    print("Для получения графиков MPL модели ------------------ 5")
    print("Для создание и обучения нейронной сети LSTM введие - 6")
    print("Для использвания нейронной сети LSTM введите ------- 7")
    print("Для получения графиков LSTM модели ----------------- 8")
    print("Для остановки работы программы введите ------------- 0")
    print("Ваш выбор:")
    flag = int(input())

    if flag == 1:
        print("------------------------------------------------")
        print("Предварительная обработка данных начата")
        handler_one = preprocessingData.PreprocessingData()
        handler_one.load_data()
        handler_one.category_definition()
        handler_one.save_all_categories(settings.path_all_category)
        handler_one.preparation_data_frame()
        handler_one.save_data_frame(settings.path_data_preprocessing)
        print("Предварительная обработка данных закончена")

    if flag == 2:
        print("------------------------------------------------")
        print("Подготовка данных к обучению начата")
        print("Для подготовки данных для модели MPL введите -- 1")
        print("Для подготовки данных для модели LSTM введите - 2")
        print("Ваш выбор:")
        q = int(input())
        handel_two = tokenizer.Tokenization(settings.path_data_preprocessing, settings.sheet_name_preprocessing)
        handel_two.quantity_categories(settings.path_all_category, settings.sheet_name_categories)
        handel_two.creation_dictionary(q)
        handel_two.data_from_arrays(train_test_split=0.9)
        if q == 1:
            handel_two.vectorization_arrays()
        if q == 2:
            handel = preprocessingData.PreprocessingData()
            max_word = handel.max_word()
            handel_two.vectorization_arrays_LSTM(max_word)
        print("Подготовка данных к обучению закончена")

    if flag == 3:
        print("------------------------------------------------")
        print("Создание и обучение нейронной сети типа MPL началось")
        handel_three = neuro_mpl.MPL()
        handel_three.load_data()
        handel_three.load_data_info()
        handel_three.create_model()
        handel_three.training_model()
        handel_three.save_model()
        print("Создание нейронной сети типа MPL закончилось")

    if flag == 4:
        print("------------------------------------------------")
        print("Использование нейронной сети начато")
        text_handler = preprocessingData.PreprocessingData()
        cat_df = text_handler.load_caregories()
        handel_four = neuro_mpl.MPL()
        handel_four.load_data_info()
        handel_four.load_model()
        dict, token = handel_four.load_token()
        request = []
        
        i = 0
        while i == 0:
            print()
            print("Введите текст для категоризации:")
            text = input()
            
            if text == "stop":
                break
            
            text = text_handler.clean_text(text)
            request.append(text)
            num_cat = handel_four.predict(dict, token, request)
            request.clear()
            result = cat_df.values[num_cat][1]
            print(u'Определённая категория: {}'.format(result))

    if flag == 5:
        print("------------------------------------------------")
        handel = neuro_mpl.MPL()
        handel.graw_plot()

    if flag == 6:
        print("------------------------------------------------")
        print("Создание и обучение нейронной сети типа LSTM началось")
        handel_six = neuro_lstm.LSTM_model()
        handel_six.load_data()
        handel_six.load_data_info()
        handel_six.create_model()
        handel_six.training_model()
        handel_six.save_model()
        print("Создание нейронной сети типа LSTM закончилось")

    if flag == 7:
        print("------------------------------------------------")
        print("Использование нейронной сети LSTM начато")
        text_handler = preprocessingData.PreprocessingData()
        cat_df = text_handler.load_caregories()
        handel_seven = neuro_lstm.LSTM_model()
        handel_seven.load_data_info()
        handel_seven.load_model()
        dict, token = handel_seven.load_token()
        request = []
        
        i = 0
        while i == 0:
            print()
            print("Введите текст для категоризации:")
            text = input()
            
            if text == "stop":
                break
            
            text = text_handler.clean_text(text)
            request.append(text)
            num_cat = handel_seven.predict(dict, token, request)
            request.clear()
            result = cat_df.values[num_cat][1]
            print(u'Определённая категория: {}'.format(result))

    if flag == 8:
        print("------------------------------------------------")
        handel = neuro_lstm.LSTM_model()
        handel.graw_plot()
    




        
