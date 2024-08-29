import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Генерация данных
np.random.seed(0)
X = np.random.uniform(-10, 10, size=(100, 1))  # Входные данные (100 точек)
y = 12 * X**2 + 3  # Выходные данные на основе функции y = 12*x^2 + 3

# Добавление небольшого шума для реальности
y += np.random.normal(0, 10, size=y.shape)

# Применение полиномиальных признаков
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Создание и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_poly, y)

# Предсказание результатов
X_test = np.linspace(-10, 10, 100).reshape(-1, 1)
X_test_poly = poly.transform(X_test)
y_pred = model.predict(X_test_poly)

# Визуализация данных и модели
plt.scatter(X, y, color='blue', label='Данные')
plt.plot(X_test, y_pred, color='red', label='Полиномиальная регрессия')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Аппроксимация функции y = 12*x^2 + 3')
plt.legend()
plt.show()

# Вывод коэффициентов модели
print("Коэффициенты модели:", model.coef_)
print("Свободный член:", model.intercept_)