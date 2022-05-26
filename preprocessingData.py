import re
import pandas
import pymorphy2
import settings

class PreprocessingData():
    """Предварительная обработка сырых данных"""

    def __init__(self):
        self.exel_data_df = None
        self.size_exel_data_df = None
        self.all_category = []
        self.quantity_category = -1
        self.result_dict = {}

    def load_data(self):
        self.exel_data_df = pandas.read_excel(settings.path_data_base, sheet_name=settings.sheet_name)
        self.size_exel_data_df = len(self.exel_data_df)
        #self.size_exel_data_df = 1000

    def clean_text(self, text):
        text = text.lower()
        text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text) # Редактирование строки
        text = re.sub('[.,:;_%©?*,!@#$%^&()\d]', ' ', text) # Удаление символов

        ma = pymorphy2.MorphAnalyzer()
        text = " ".join(ma.parse(str(word))[0].normal_form for word in text.split()) # Слова к нормальному виду

        text = ' '.join(word for word in text.split() if len(word)>3) # Убираем слова которые меньше 3 символов
        #text = text.encode("utf-8")

        return text
    
    def category_definition(self):
        self.all_category.append(self.exel_data_df.values[0][3])

        i = 1
        while i < self.size_exel_data_df:
            flag = 0
            j = 0
            while j < len(self.all_category):
                print(i)
                print(j)
                print('---')
            
                if self.exel_data_df.values[i][3] == self.all_category[j]:
                    flag = 1
                    break

                j = j + 1
        
            if flag == 0:
                self.all_category.append(self.exel_data_df.values[i][3])
            i = i + 1
        
        self.quantity_category = len(self.all_category)
    
    def save_all_categories(self, path_file):
        all_cat_dict = {}
        all_cat_dict['Категории'] = self.all_category
        data_frame = pandas.DataFrame(all_cat_dict)
        data_frame.to_excel(path_file)
    
    def find_category(self, text):
        i = 0

        while i < self.quantity_category:
            if text == self.all_category[i]:
                return i
            i = i + 1
        return -1

    def preparation_data_frame(self):
        request = []
        caregory = []

        i = 0
        while i < self.size_exel_data_df:
            request.append(self.clean_text(self.exel_data_df.values[i][0]))
            print('-')
            caregory.append(self.find_category(self.exel_data_df.values[i][3]))
            print('--')
            i = i + 1
            print(i)

        self.result_dict['Запрос'] = request
        self.result_dict['Категория'] = caregory

    def save_data_frame(self, path_file):
        data_frame = pandas.DataFrame(self.result_dict)
        data_frame.to_excel(path_file)
    
    def load_caregories(self):
        cat_df = pandas.read_excel(settings.path_all_category, sheet_name=settings.sheet_name_categories)
        return cat_df
    
    def max_word(self):
        max_words = 0
        data_df = pandas.read_excel(settings.path_data_preprocessing, sheet_name=settings.sheet_name_preprocessing)
        text = data_df['Запрос']
        for desc in text:
            words = len(desc.split())
            if words > max_words:
                max_words = words
        
        print('Максимальная длина описания: {} слов'.format(max_words))
        
        return max_words


        
        
        
    

    
