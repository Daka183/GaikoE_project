# Путь к сырым данным
path_data_base = 'popisyat.xls'
# Название листа
sheet_name = 'Лист1'
# Путь к данным для тренировки
path_data_preprocessing = 'result.xlsx'
# Название листа для result
sheet_name_preprocessing = 'Sheet1'
# Путь к данным о всех категориях
path_all_category = 'all_category.xlsx'
# Название листа для категорий
sheet_name_categories = 'Sheet1'

verbose = 1     # Анимация при обучении

# Параметры для обучения нейронной сети MPL
epochs_mpl = 10                 # Количество эпох
batch_size_mpl = 32             # Количество образцов на обновление градиента.
validation_split_mpl = 0.25     # Часть данных об обучении, которая будет использоваться в качестве данных для проверки

# Функция потерь
loss_mpl = 'categorical_crossentropy'
# Оптимизатор
optimizer_mpl = 'adam'

# Параметры для обучения нейронной сети LSTM
epochs_lstm = 5                 # Количество эпох
batch_size_lstm = 32            # Количество образцов на обновление градиента.

# Функция потерь
loss_lstm = 'binary_crossentropy'
# Оптимизатор
optimizer_lstm = 'adam'

# Доступны функции активации активации:
# elu - Экспоненциальный линейный блок
# softmax - Функция активации Softmax
# selu - Масштабируемая экспоненциальная линейная единица (SELU)
# softplus - Функция активации Softplus
# softsign - Функция активации softsign
# relu - Спрямленный линейный блок
# tanh - Функция активации гиперболического тангенса
# sigmoid - Функция активации сигмоида
# hard_sigmoid - Функция активации жествкая сигмоида. Быстрее вычисляет, чем активацию сигмоида.
# exponential - Экспоненциальная активация: exp(x).
# linear - Функция линейной (т.е. идентификационная) активации

# Функции потерь:
# categorical_crossentropy
# binary_crossentropy
# mean_squared_error
# mean_absolute_error
# mean_absolute_percentage_error
# mean_squared_logarithmic_error
# squared_hinge
# hinge
# categorical_hinge
# logcosh
# huber_loss
# kullback_leibler_divergence
# poisson
# cosine_proximity

# Доступные оптимизаторы:
# SGD
# RMSprop
# Adagrad
# Adadelta
# Adam
# Adamax
# Nadam

