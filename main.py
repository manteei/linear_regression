import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
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

numeric_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']

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

#Реализуйте линейную регрессию с использованием метода наименьших квадратов
def linear_regression(X, y):
    # Добавляем столбец с единицами для учёта свободного члена (w0)
    X = np.column_stack((np.ones(X.shape[0]), X))
    # Вычисляем оптимальные веса с использованием формулы
    weights = np.linalg.inv(X.T @ X) @ X.T @ y
    return weights

# Обучаем модель на обучающих данных
optimal_weights = linear_regression(X_train.values, y_train.values)

# Функция для предсказания на тестовых данных
def predict(X, weights):
    # Добавляем столбец с единицами для учёта свободного члена (w0)
    X = np.column_stack((np.ones(X.shape[0]), X))
    # Предсказываем значения
    predictions = X @ weights
    return predictions

# Получаем предсказания на тестовых данных
y_pred = predict(X_test.values, optimal_weights)

# Вычисляем среднеквадратичную ошибку (MSE) на тестовых данных
mse = np.mean((y_test.values - y_pred) ** 2)

# Выводим оптимальные веса и MSE
print("Оптимальные коэффициенты:", optimal_weights)
print("Среднеквадратичная ошибка на тестовых данных:", mse)

# Определите новые наборы признаков
features_set1 = ['longitude', 'latitude', 'housing_median_age', 'median_income']
features_set2 = ['total_rooms', 'total_bedrooms', 'households']
features_set3 = ['population', 'median_income']

# Разделите данные на обучающий и тестовый наборы данных для каждого набора признаков
X_train_set1, X_test_set1, y_train, y_test = train_test_split(df[features_set1], y, test_size=0.2, random_state=42)
X_train_set2, X_test_set2, y_train, y_test = train_test_split(df[features_set2], y, test_size=0.2, random_state=42)
X_train_set3, X_test_set3, y_train, y_test = train_test_split(df[features_set3], y, test_size=0.2, random_state=42)

# Обучите модели на каждом наборе признаков
optimal_weights_set1 = linear_regression(X_train_set1.values, y_train.values)
optimal_weights_set2 = linear_regression(X_train_set2.values, y_train.values)
optimal_weights_set3 = linear_regression(X_train_set3.values, y_train.values)

# Получите предсказания для каждой модели
y_pred_set1 = predict(X_test_set1.values, optimal_weights_set1)
y_pred_set2 = predict(X_test_set2.values, optimal_weights_set2)
y_pred_set3 = predict(X_test_set3.values, optimal_weights_set3)
np.set_printoptions(threshold=np.inf)

print("Предсказанные значения для модели 1:", y_pred_set1)
print("Предсказанные значения для модели 2:", y_pred_set2)
print("Предсказанные значения для модели 3:", y_pred_set3)


# Оцените качество моделей, например, с помощью MSE
mse_set1 = mean_squared_error(y_test, y_pred_set1)
mse_set2 = mean_squared_error(y_test, y_pred_set2)
mse_set3 = mean_squared_error(y_test, y_pred_set3)

print("MSE для модели 1:", mse_set1)
print("MSE для модели 2:", mse_set2)
print("MSE для модели 3:", mse_set3)

r2_set1 = r2_score(y_test, y_pred_set1)
r2_set2 = r2_score(y_test, y_pred_set2)
r2_set3 = r2_score(y_test, y_pred_set3)

print("Коэффициент детерминации для модели 1:", r2_set1)
print("Коэффициент детерминации для модели 2:", r2_set2)
print("Коэффициент детерминации для модели 3:", r2_set3)