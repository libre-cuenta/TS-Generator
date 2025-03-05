from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.graphics.gofplots import ProbPlot


class Korezin_TS(ABC):
    def __init__(self, data):
        """
        Инициализация модели SARIMA
        order: кортеж (p,d,q) - параметры несезонной части
        seasonal_order: кортеж (P,D,Q,s) - параметры сезонной части
        """
        self.data = data
        self.train = None
        self.test = None
        self.order = (1,1,1)
        self.aic = None
        self.model = None
        self.residuals = None
    
    def train_test_split_ts(self, test_size=0.2):
        """
        Разделение временного ряда на обучающую и тестовую выборки
        """
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data)

        train_size = int(len(self.data) * (1 - test_size))
        self.train, self.test = self.data[:train_size], self.data[train_size:]
        return self
    
    def check_stationarity(self):
        """
        Проверка стационарности временного ряда с помощью теста Дики-Фуллера
        """
        result = adfuller(self.data)
        print('ADF Statistic:', result[0])
        print('p-value:', result[1])
        print('Critical values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
        
        # Если p-value > 0.05, ряд нестационарен
        return (result[1] <= 0.05)
    
    def find_optimal_d(self, max_d=2):
        """
        Определение оптимального параметра d путем последовательного дифференцирования
        """
        d = 0
        series = self.data.copy()
        
        while d <= max_d:
            if self.check_stationarity():
                self.order = (self.order[0], d, self.order[2])
                print(f'Optional d: {d}')
                return d
            series = np.diff(series)
            d += 1
        
        self.order = (self.order[0], d, self.order[2])
        print(f'Optional d: {d}')
        return self

    def predict(self, steps):
        """
        Построение прогноза на n_steps шагов вперед
        """
        forecast = self.model.fit().forecast(steps=steps)
        return forecast
    
    def evaluate_predictions(self, actual, predicted):
        """
        Оценка качества прогноза
        """
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    def suggest_model_improvements(self):
        # Проверка автокорреляции
        acf_values = acf(self.residuals)
        
        if abs(acf_values[1]) > 0.2:
            print("Рекомендуется увеличить порядок AR")
        else:
            print("Порядок AR соответствует модели")
        
        if abs(acf_values[-1]) > 0.2:
            print("Рекомендуется увеличить порядок MA")
        else:
            print("Порядок MA соответствует модели")

    @abstractmethod
    def fit_model(self):
        pass

    # @abstractmethod
    # def predict_by_step(self, step, predicted_view):
    #     pass

    def plot_diagnostics(self):
        """
        Построение диагностических графиков
        """
        fig = self.model.fit().plot_diagnostics(figsize=(15, 12))
        plt.show()

    @abstractmethod
    def plot_predictions(self, predictions):
        pass

    def plot_residuals(self):
        """
        Анализ остатков модели
        """
        residuals = pd.DataFrame(self.residuals)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # График остатков
        residuals.plot(title="Остатки", ax=ax1)
        ax1.set_xlabel("Время")
        ax1.set_ylabel("Значение остатков")
        
        # Гистограмма остатков
        residuals.plot(kind='kde', title="Плотность распределения остатков", ax=ax2)
        
        # Q-Q график
        QQ = ProbPlot(residuals)
        QQ.qqplot(line='45', ax=ax3)
        ax3.set_title("Q-Q График")
        
        # Автокорреляция остатков
        acf_values = acf(residuals.values.squeeze(), nlags=40)
        ax4.plot(range(len(acf_values)), acf_values)
        ax4.set_title("Автокорреляция остатков")
        ax4.axhline(y=0, linestyle='--', color='gray')
        ax4.axhline(y=-1.96/np.sqrt(len(residuals)), linestyle='--', color='gray')
        ax4.axhline(y=1.96/np.sqrt(len(residuals)), linestyle='--', color='gray')
        
        plt.tight_layout()
        plt.show()


class ARIMA_model(Korezin_TS):
    def __init__(self, data):
        """
        Инициализация модели SARIMA
        order: кортеж (p,d,q) - параметры несезонной части
        """
        super().__init__(data)

    def find_best_arima_params(self, max_p, max_q):
        """
        Подбор оптимальных параметров p, d, q для модели ARIMA
        """
        best_aic = float('inf')
        best_params = self.order
        
        # Перебираем все возможные комбинации параметров p, d и q
        for p, q in product(range(max_p + 1), range(max_q + 1)):
            if p == 0 and q == 0:
                continue
                
            try:
                model = ARIMA(self.data, order=(p, self.order[1], q))
                results = model.fit()
                print(results.aic)
                    
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = (p, self.order[1], q)
                        
            except:
                continue
        
        self.order = best_params
        self.aic = best_aic
        return self

    def fit_model(self):
        """
        Обучение модели ARMA с заданными параметрами
        """
        self.model = ARIMA(self.train, order=self.order)  # ARIMA(p,d,q)
        self.residuals = self.model.fit().resid
        return self


    def plot_predictions(self, predictions):
        """
        Визуализация результатов прогнозирования
        """
        title='ARIMA Прогноз'
        
        plt.figure(figsize=(12, 6))
        
        # Построение обучающих данных
        plt.plot(range(len(self.train)), self.train, 
                label='Обучающие данные', color='#ff0000')
        
        # Построение тестовых данных
        plt.plot(range(len(self.train), len(self.train) + len(self.test)), 
                self.test, label='Тестовые данные', color='#00ff00')
        
        # Построение прогноза
        plt.plot(range(len(self.train), len(self.train) + len(predictions)), 
                predictions, label='Прогноз', color='#0000ff', linestyle='--')
        
        plt.title(title)
        plt.xlabel('Время')
        plt.ylabel('Значение')
        plt.legend()
        plt.grid(True)
        plt.show()


class SARIMA_model(Korezin_TS):
    def __init__(self, data):
        """
        Инициализация модели SARIMA
        order: кортеж (p,d,q) - параметры несезонной части
        seasonal_order: кортеж (P,D,Q,s) - параметры сезонной части
        """
        super().__init__(data)
        self.seasonal_order = (1,1,1,12)


    def find_optimal_D(self, season_length, max_D=2):
        """
        Определение оптимального сезонного параметра D
        """
        D = 0
        series = self.data.copy()
        while D <= max_D:
            if self.check_stationarity():
                self.seasonal_order = (self.seasonal_order[0], D, self.seasonal_order[2], self.seasonal_order[3])
                print(f'Optional D: {D}')
                return D
            series = np.diff(series, n=season_length)
            D += 1
        self.seasonal_order = (self.seasonal_order[0], D, self.seasonal_order[2], self.seasonal_order[3])
        return self

    def find_best_sarima_params(self, season_length,
                        max_p, max_q,
                        max_P, max_Q):
        """
        Поиск оптимальных параметров SARIMA по сетке
        """
        best_aic = float('inf')
        best_params = self.order
        
        for p, q, P, Q in product(range(max_p + 1), range(max_q + 1), range(max_P + 1), range(max_Q + 1)):
            # Формируем параметры модели
            order = (p, self.order[1], q)
            seasonal_order = (P, self.seasonal_order[1], Q, season_length)
            
            try:
                model = SARIMAX(self.data,
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
                results = model.fit(disp=False)
                
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = (order, seasonal_order)
                    
                print(f'SARIMA{order}x{seasonal_order} - AIC:{results.aic:.3f}')
                
            except:
                continue
                
        self.order = best_params[0]
        self.seasonal_order = best_params[1]
        self.aic = best_aic
        return self

    def fit_model(self):
        """
        Обучение модели SARIMA с заданными параметрами
        """
        self.model = SARIMAX(self.train, order=self.order, seasonal_order=self.seasonal_order)  # ARIMA(p,d,q)
        self.residuals = self.model.fit().resid
        return self


    def plot_predictions(self, predictions):
        """
        Визуализация результатов прогнозирования
        """
        title='SARIMA Прогноз'
        
        plt.figure(figsize=(12, 6))
        
        # Построение обучающих данных
        plt.plot(range(len(self.train)), self.train, 
                label='Обучающие данные', color='#ff0000')
        
        # Построение тестовых данных
        plt.plot(range(len(self.train), len(self.train) + len(self.test)), 
                self.test, label='Тестовые данные', color='#00ff00')
        
        # Построение прогноза
        plt.plot(range(len(self.train), len(self.train) + len(predictions)), 
                predictions, label='Прогноз', color='#0000ff', linestyle='--')
        
        plt.title(title)
        plt.xlabel('Время')
        plt.ylabel('Значение')
        plt.legend()
        plt.grid(True)
        plt.show()



class ARCH_model:
    def __init__(self, data):
        """
        Инициализация модели SARIMA
        order: кортеж (p,d,q) - параметры несезонной части
        seasonal_order: кортеж (P,D,Q,s) - параметры сезонной части
        """
        self.data = data
        self.train = None
        self.test = None
        self.order = (1,1)
        self.aic = None
        self.model = None
        self.residuals = None

    def train_test_split_ts(self, test_size=0.2):
        """
        Разделение временного ряда на обучающую и тестовую выборки
        """
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data)

        train_size = int(len(self.data) * (1 - test_size))
        self.train, self.test = self.data[:train_size], self.data[train_size:]
        return self

    def check_stationarity(self):
            """
            Проверка стационарности временного ряда с помощью теста Дики-Фуллера
            """
            result = adfuller(self.data)
            print('ADF Statistic:', result[0])
            print('p-value:', result[1])
            print('Critical values:')
            for key, value in result[4].items():
                print('\t%s: %.3f' % (key, value))
            
            # Если p-value > 0.05, ряд нестационарен
            return (result[1] <= 0.05)

    # Функция для построения и обучения модели ARCH
    def fit_arch_model(self):
        self.model = arch_model(self.data, vol='ARCH', p=self.order[0], q=self.order[1])
        return self
    
    def fit_garch_model(self):
        self.model = arch_model(self.data, vol='GARCH', p=self.order[0], q=self.order[1])
        return self

    # Добавление кросс-валидации
    def cross_validate_model(self, returns, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        rmse_scores = []
        
        for train_index, test_index in tscv.split(returns):
            train_returns = returns[train_index]
            test_returns = returns[test_index]
            
            predicted_vol = np.sqrt(self.model.fit().forecast(horizon=len(test_returns)).variance.values[-1, :])
            actual_vol = np.abs(test_returns.values)
            
            rmse = np.sqrt(mean_squared_error(actual_vol, predicted_vol[:len(actual_vol)]))
            rmse_scores.append(rmse)
        
        return np.mean(rmse_scores), np.std(rmse_scores)
    
    # Добавление оптимизации гиперпараметров
    def optimize_arch_parameters(self, max_p, max_q):
        best_aic = float('inf')
        best_params = self.order
        
        for p, q in product(range(max_p + 1), range(max_q + 1)):
            if p == 0 and q == 0:
                continue
                
            try:
                model = arch_model(self.data, vol='Garch', p=p, q=q)
                results = model.fit(disp='off')
                print(results.aic)
                    
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = (p, q)
                        
            except:
                continue
        
        self.order = best_params
        self.aic = best_aic
        return self

    def historical_volatility(self):
        return np.sqrt(self.model.fit().conditional_volatility)

    # Функция для получения прогноза волатильности
    def predict(self, steps):
        forecast = self.model.fit().forecast(horizon=steps)
        return np.sqrt(forecast.variance.values[-1, :])

    def evaluate_predictions(self, actual, predicted):
        predicted_vol = np.sqrt(self.model.fit().forecast(horizon=len(self.test)).variance.values[-1, :])
        actual_vol = np.abs(self.test)
        rmse = np.sqrt(mean_squared_error(actual_vol, predicted_vol[:len(actual_vol)]))
        return {
            'RMSE': rmse
        }
    
    # Функция для визуализации результатов
    def plot_results(self, volatility, forecast):
        plt.figure(figsize=(12, 6))
        
        # Построение графика доходности
        plt.subplot(2, 1, 1)
        plt.plot(self.data, label='Returns')
        plt.title('Returns and Volatility')
        plt.legend()
        
        # Построение графика волатильности
        plt.subplot(2, 1, 2)
        plt.plot(volatility, label='Historical Volatility')
        plt.plot(range(len(volatility), len(volatility) + len(forecast)), 
                forecast, label='Forecasted Volatility')
        plt.legend()
        
        plt.tight_layout()
        plt.show()



class ARIMAGARCHPredictor:
    def __init__(self, arima_model, garch_model):
        self.arima_model = arima_model
        self.garch_model = garch_model

    def fit(self, arima_model, garch_model):
        """
        Импорт моделей ARIMA и GARCH
        """
        self.arima_model = arima_model
        self.garch_model = garch_model
        
    def predict(self, steps):
        """
        Прогнозирование на заданное количество шагов вперед
        """
        # Прогноз ARIMA
        arima_forecast = self.arima_model.fit().forecast(steps=steps)
        
        # Прогноз волатильности GARCH
        garch_forecast = self.garch_model.fit().forecast(horizon=steps)
        volatility = np.sqrt(garch_forecast.variance.values[-1, :])
        
        # Комбинированный прогноз
        forecast = pd.DataFrame({
            'mean': arima_forecast,
            'volatility': volatility,
            'lower_bound': arima_forecast - 1.96 * volatility,
            'upper_bound': arima_forecast + 1.96 * volatility
        })
        
        return forecast

    def evaluate_model(self, predictions, actual):
        """
        Оценка качества модели
        """
        mse = mean_squared_error(actual, predictions['mean'])
        rmse = np.sqrt(mse)
        
        # Проверка попадания в доверительный интервал
        in_interval = np.sum((actual >= predictions['lower_bound']) & 
                            (actual <= predictions['upper_bound']))
        interval_coverage = in_interval / len(actual)
        
        return {
            'rmse': rmse,
            'interval_coverage': interval_coverage
        }

    def plot_results(self, actual, predictions, title='ARIMA+GARCH Forecast'):
        """
        Визуализация результатов
        """
        plt.figure(figsize=(12, 6))
        
        # Построение фактических значений
        plt.plot(actual, label='Actual', color='blue')
        
        # Построение прогноза
        plt.plot(predictions['mean'], label='Forecast', color='red')
        
        # Построение доверительного интервала
        plt.fill_between(
            predictions.index,
            predictions['lower_bound'],
            predictions['upper_bound'],
            color='gray',
            alpha=0.2,
            label='95% Confidence Interval'
        )
        
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
