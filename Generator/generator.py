import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numexpr as ne

class TSGenerator:
    @classmethod
    def generate_by_expression(cls, time, expr, noise_std = 1):
        # expr - выражение с параметром t
        # time - значения параметра t
        # noise_std - стандартное отклонение шума

        series = list()
        
        for t in time:
            series.append(ne.evaluate(expr))
        
        noise = np.random.normal(0, noise_std, len(time))
        series += noise
        
        return series

    @classmethod
    def generate_ar_series(cls, n, ar_params, noise_std=1):
        """
        Генерация временного ряда на основе модели AR(p).

        :param n: Длина временного ряда
        :param ar_params: Коэффициенты авторегрессии (список)
        :param noise_std: Стандартное отклонение шума
        :return: Временной ряд (numpy массив)
        """
        p = len(ar_params)
        series = np.zeros(n)
        noise = np.random.normal(0, noise_std, n)

        for t in range(p, n):
            series[t] = np.dot(ar_params, series[t-p:t][::-1]) + noise[t]

        return series

    @classmethod
    def generate_ma_series(cls, n, ma_params, noise_std=1):
        """
        Генерация временного ряда на основе модели MA(q).

        :param n: Длина временного ряда
        :param ma_params: Коэффициенты скользящего среднего (список)
        :param noise_std: Стандартное отклонение шума
        :return: Временной ряд (numpy массив)
        """
        q = len(ma_params)
        series = np.zeros(n)
        noise = np.random.normal(0, noise_std, n)

        for t in range(n):
            # Добавление значения MA(q)
            if t == 0:
                series[t] = noise[t]  # Первое значение - это только шум
            else:
                for j in range(1, min(q, t) + 1):
                    series[t] += ma_params[j - 1] * noise[t - j]
                series[t] += noise[t]  # Добавляем текущее значение шума

        return series

    @classmethod
    def generate_arma_series(cls, n, ar_params, ma_params, noise_std=1):

        """
        Генерация временного ряда на основе модели ARMA(p, q).

        :param n: Длина временного ряда
        :param ar_params: Коэффициенты авторегрессии (список)
        :param ma_params: Коэффициенты скользящего среднего (список)
        :param noise_std: Стандартное отклонение шума
        :return: Временной ряд (numpy массив)
        """

        p = len(ar_params)
        q = len(ma_params)
        series = np.zeros(n)
        noise = np.random.normal(0, noise_std, n)

        # Генерация временного ряда
        for t in range(max(p, q) + 1, n):
            # AR часть
            ar_term = sum(ar_params[j] * series[t - j - 1] for j in range(p))
            # MA часть
            ma_term = sum(ma_params[j] * noise[t - j - 1] for j in range(q))
            series[t] = ar_term + ma_term + noise[t]

        return series

    @classmethod
    def generate_arima_series(cls, n, ar_params, d, ma_params, noise_std=1):
        """
        Генерация временного ряда на основе модели ARIMA(p, d, q).

        :param n: Длина временного ряда
        :param ar_params: Коэффициенты авторегрессии (список)
        :param d: Порядок дифференцирования
        :param ma_params: Коэффициенты скользящего среднего (список)
        :param noise_std: Стандартное отклонение шума
        :return: Временной ряд (numpy массив)
        """
        p = len(ar_params)
        q = len(ma_params)
        
        # Шум, создаваемый нормальным распределением
        noise = np.random.normal(0, noise_std, n)
        
        # Создаем временной ряд с нуля
        series = np.zeros(n)

        # Применяем разности (различие)
        for i in range(d):
            series[i] = noise[i] if i < n else 0  # Заполнение первых d значений

        for t in range(max(p, q), n):
            # AR часть
            ar_term = sum(ar_params[j] * series[t - j - 1] for j in range(p))
            
            # MA часть
            ma_term = sum(ma_params[j] * noise[t - j - 1] for j in range(q))

            series[t] = ar_term + ma_term + noise[t]

        # Возвращаем временной ряд с интегрированием
        for i in range(d):
            series = np.cumsum(series) # Интеграция путем кумуляции

        return series

    @classmethod
    def generate_sarimax_series(cls, n, ar_params, d, ma_params, seasonal_order, noise_std=1):
        """
        Генерация временного ряда на основе модели SARIMAX.

        :param n: Длина временного ряда
        :param ar_params: Коэффициенты авторегрессии (список)
        :param d: Порядок дифференцирования
        :param ma_params: Коэффициенты скользящего среднего (список)
        :param seasonal_order: Параметры сезонности (p, d, q, s)
        :param noise_std: Стандартное отклонение шума
        :return: Временной ряд (numpy массив)
        """
        p = len(ar_params)
        q = len(ma_params)
        s = seasonal_order[3]  # Длина сезонного цикла
        
        # Генерация нормального шума
        noise = np.random.normal(0, noise_std, n)

        # Генерация временного ряда
        series = np.zeros(n)

        # Применяем разности (различие)
        for _ in range(d):
            series = np.diff(series, n=1, prepend=noise[0])

        for t in range(max(p, q, seasonal_order[0] + seasonal_order[2]), n):
            # AR часть
            ar_term = sum(ar_params[j] * series[t - j - 1] for j in range(p))
            # MA часть
            ma_term = sum(ma_params[j] * noise[t - j - 1] for j in range(q))
            
            # Сезонные члены
            seasonal_ar = sum(seasonal_order[0] * series[t - s - j] for j in range(seasonal_order[0]))
            seasonal_ma = sum(seasonal_order[2] * noise[t - s - j] for j in range(seasonal_order[2]))
            
            # Генерация следующего значения
            series[t] = ar_term + ma_term + seasonal_ar + seasonal_ma + noise[t]

        return series

