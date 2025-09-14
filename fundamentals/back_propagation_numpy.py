import numpy as np


def f(x):
    """Функция активации - гиперболический тангенс"""
    return 2 / (1 + np.exp(-x)) - 1


def df(x):
    """Производная от функции активации"""
    return 0.5 * (1 + x) * (1 - x)


w1 = np.array([[-0.2, 0.3, -0.4], [0.1, -0.3, -0.4]])  # начальные веса для нейронов первого слоя
w2 = np.array([0.2, 0.3])  # начальные веса для нейронов второго слоя


def go_forward(input):
    first_layer_sum = np.dot(w1, input)
    first_layer_out = np.array([f(x) for x in first_layer_sum])  # [f11, f12]

    second_layer_sum = np.dot(w2, first_layer_out)
    second_layer_out = np.array(f(second_layer_sum))  # y

    return first_layer_out, second_layer_out


def train(data, lmd=0.01, n_iter=10_000):
    global w1, w2
    len_data = len(data)

    for k in range(n_iter):
        to_train = data[np.random.randint(0, len_data)]  # случайный выбор входных данных из обучающей выборки
        out, y = go_forward(to_train[:-1])  # проход по нс и вычисление значений нейронов на каждом слое

        e = y - to_train[-1]  # вычисление ошибки
        delta = e * df(y)  # вычисление локального градиента

        w2[0] = w2[0] - lmd * delta * out[0]  # корректировка веса первой связи
        w2[1] = w2[1] - lmd * delta * out[1]  # корректировка веса второй связи

        delta2 = w2 * delta * df(out)  # вектор из 2-х величин локальных градиентов

        # корректировка связей первого слоя
        w1[0, :] = w1[0, :] - np.array(to_train[:-1]) * delta2[0] * lmd
        w1[1, :] = w1[1, :] - np.array(to_train[:-1]) * delta2[1] * lmd


# обучающая выборка (она же полная выборка)
dataset = [(-1, -1, -1, -1),
           (-1, -1, 1, 1),
           (-1, 1, -1, -1),
           (-1, 1, 1, 1),
           (1, -1, -1, -1),
           (1, -1, 1, 1),
           (1, 1, -1, -1),
           (1, 1, 1, -1)]

train(dataset, lmd=0.01, n_iter=10_000)

for x in dataset:
    out, y = go_forward(x[:-1])
    print(f"Выходное значение НС: {y} => {x[-1]}")
