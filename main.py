import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from visualize import visCount, visMean, visAll, visStd, visMin, visMax

# Получите статистику по датасету
pd.set_option('display.max_columns', None)
df = pd.read_csv('california_housing_train.csv')
summary_stats = df.describe()
""""
#Визуализируйте статистику по датасету
visAll(df)
# Визуализация количества
visCount(plt, df)
# Визуализация средних значений
visMean(plt, df)
# Визуализация стандартных отклонений
visStd(plt, df)
# Визуализация минимума
visMin(plt, df)
# Визуализация максимума
visMax(plt, df)
"""

# Проведите предварительную обработку данных
df.fillna(df.mean(), inplace=True)

numeric_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']

# Примените стандартизацию к каждому числовому признаку
for column in numeric_columns:
    mean = df[column].mean()
    std = df[column].std()
    df[column] = (df[column] - mean) / std

# Определите признаки (X) и целевую переменную (y)
X = df.drop(columns=['median_house_value'])  # Признаки, исключая 'median_house_value'
y = df['median_house_value']  # Целевая переменная 'median_house_value'
#Разделите данные на обучающий и тестовый наборы данных

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)