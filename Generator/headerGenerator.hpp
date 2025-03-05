#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <functional>
#include <cmath>

// Helper function for cumulative sum (integration)
static std::vector<double> cumsum(const std::vector<double>& input);

// Helper function to mimic numpy.diff with prepend value
static std::vector<double> diff(const std::vector<double>& input, double prepend_val);


class TSGenerator {
    public:
        // Генерация временного ряда на основе модели AR(p).
        // 
        // :param n: Длина временного ряда
        // :param ar_params: Коэффициенты авторегрессии (список)
        // :param noise_std: Стандартное отклонение шума
        // :return: Временной ряд (numpy массив)
        static std::vector<double> generate_ar_series(int n, const std::vector<double>& ar_params, double noise_std = 1);
    
        // Генерация временного ряда на основе модели MA(q).
        //
        // :param n: Длина временного ряда
        // :param ma_params: Коэффициенты скользящего среднего (список)
        // :param noise_std: Стандартное отклонение шума
        // :return: Временной ряд (numpy массив)
        static std::vector<double> generate_ma_series(int n, const std::vector<double>& ma_params, double noise_std = 1);
    
        // Генерация временного ряда на основе модели ARMA(p, q).
        //
        // :param n: Длина временного ряда
        // :param ar_params: Коэффициенты авторегрессии (список)
        // :param ma_params: Коэффициенты скользящего среднего (список)
        // :param noise_std: Стандартное отклонение шума
        // :return: Временной ряд (numpy массив)
        static std::vector<double> generate_arma_series(int n, const std::vector<double>& ar_params, const std::vector<double>& ma_params, double noise_std = 1);
    
        // Генерация временного ряда на основе модели ARIMA(p, d, q).
        //
        // :param n: Длина временного ряда
        // :param ar_params: Коэффициенты авторегрессии (список)
        // :param d: Порядок дифференцирования
        // :param ma_params: Коэффициенты скользящего среднего (список)
        // :param noise_std: Стандартное отклонение шума
        // :return: Временной ряд (numpy массив)
        static std::vector<double> generate_arima_series(int n, const std::vector<double>& ar_params, int d, const std::vector<double>& ma_params, double noise_std = 1);
    
        // Генерация временного ряда на основе модели SARIMAX.
        //
        // :param n: Длина временного ряда
        // :param ar_params: Коэффициенты авторегрессии (список)
        // :param d: Порядок дифференцирования
        // :param ma_params: Коэффициенты скользящего среднего (список)
        // :param seasonal_order: Параметры сезонности (p, d, q, s)
        // :param noise_std: Стандартное отклонение шума
        // :return: Временной ряд (numpy массив)
        static std::vector<double> generate_sarimax_series(int n, const std::vector<double>& ar_params, int d, const std::vector<double>& ma_params, const std::vector<int>& seasonal_order, double noise_std = 1);
        
    };
    