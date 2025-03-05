#include "headerGenerator.hpp"

// Helper function for cumulative sum (integration)
static std::vector<double> cumsum(const std::vector<double>& input) {
    std::vector<double> output(input.size(), 0.0);
    if (input.empty()) {
        return output;
    }
    output[0] = input[0];
    for (size_t i = 1; i < input.size(); ++i) {
        output[i] = output[i - 1] + input[i];
    }
    return output;
}

// Helper function to mimic numpy.diff with prepend value
static std::vector<double> diff(const std::vector<double>& input, double prepend_val) {
    std::vector<double> output(input.size(), 0.0);
    if (input.empty()) {
        return output;
    }
    output[0] = input[0] - prepend_val;
    for (size_t i = 1; i < input.size(); ++i) {
        output[i] = input[i] - input[i - 1];
    }
    return output;
}



std::vector<double> TSGenerator::generate_ar_series(int n, const std::vector<double>& ar_params, double noise_std = 1) {
    int p = static_cast<int>(ar_params.size());
    std::vector<double> series(n, 0.0);
    std::vector<double> noise(n, 0.0);
            
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, noise_std);
    for (int i = 0; i < n; ++i) {
        noise[i] = dist(gen);
    }
            
    for (int t = p; t < n; ++t) {
        double dot_product = 0.0;
        for (int j = 0; j < p; ++j) {
        // series[t-p:t][::-1] means reverse order: series[t - 1], series[t - 2], ..., series[t - p]
            dot_product += ar_params[j] * series[t - j - 1];
        }
        series[t] = dot_product + noise[t];
    }
            
    return series;
}
    
        // Генерация временного ряда на основе модели MA(q).
        //
        // :param n: Длина временного ряда
        // :param ma_params: Коэффициенты скользящего среднего (список)
        // :param noise_std: Стандартное отклонение шума
        // :return: Временной ряд (numpy массив)
std::vector<double> TSGenerator::generate_ma_series(int n, const std::vector<double>& ma_params, double noise_std = 1) {
    int q = static_cast<int>(ma_params.size());
    std::vector<double> series(n, 0.0);
     std::vector<double> noise(n, 0.0);
            
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, noise_std);
    for (int i = 0; i < n; ++i) {
        noise[i] = dist(gen);
    }
            
    for (int t = 0; t < n; ++t) {
        // Добавление значения MA(q)
        if (t == 0) {
            series[t] = noise[t];  // Первое значение - это только шум
        } else {
            for (int j = 1; j <= std::min(q, t); ++j) {
                series[t] += ma_params[j - 1] * noise[t - j];
            }
            series[t] += noise[t];  // Добавляем текущее значение шума
        }
    }
            
    return series;
}
    
        // Генерация временного ряда на основе модели ARMA(p, q).
        //
        // :param n: Длина временного ряда
        // :param ar_params: Коэффициенты авторегрессии (список)
        // :param ma_params: Коэффициенты скользящего среднего (список)
        // :param noise_std: Стандартное отклонение шума
        // :return: Временной ряд (numpy массив)
std::vector<double> TSGenerator::generate_arma_series(int n, const std::vector<double>& ar_params, const std::vector<double>& ma_params, double noise_std = 1) {
    int p = static_cast<int>(ar_params.size());
    int q = static_cast<int>(ma_params.size());
    std::vector<double> series(n, 0.0);
    std::vector<double> noise(n, 0.0);
            
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, noise_std);
    for (int i = 0; i < n; ++i) {
        noise[i] = dist(gen);
    }
            
    int start = std::max(p, q) + 1;
    for (int t = start; t < n; ++t) {
        // AR часть
        double ar_term = 0.0;
        for (int j = 0; j < p; ++j) {
            ar_term += ar_params[j] * series[t - j - 1];
        }
        // MA часть
        double ma_term = 0.0;
        for (int j = 0; j < q; ++j) {
            ma_term += ma_params[j] * noise[t - j - 1];
        }
        series[t] = ar_term + ma_term + noise[t];
    }
            
    return series;
}
    
        // Генерация временного ряда на основе модели ARIMA(p, d, q).
        //
        // :param n: Длина временного ряда
        // :param ar_params: Коэффициенты авторегрессии (список)
        // :param d: Порядок дифференцирования
        // :param ma_params: Коэффициенты скользящего среднего (список)
        // :param noise_std: Стандартное отклонение шума
        // :return: Временной ряд (numpy массив)
std::vector<double> TSGenerator::generate_arima_series(int n, const std::vector<double>& ar_params, int d, const std::vector<double>& ma_params, double noise_std = 1) {
    int p = static_cast<int>(ar_params.size());
    int q = static_cast<int>(ma_params.size());
            
    // Шум, создаваемый нормальным распределением
    std::vector<double> noise(n, 0.0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, noise_std);
    for (int i = 0; i < n; ++i) {
        noise[i] = dist(gen);
    }
            
    // Создаем временной ряд с нуля
    std::vector<double> series(n, 0.0);
            
    // Применяем разности (различие)
    for (int i = 0; i < d; ++i) {
        if (i < n) {
            series[i] = noise[i];  // Заполнение первых d значений
        } else {
            series[i] = 0;
        }
    }
            
    for (int t = std::max(p, q); t < n; ++t) {
        // AR часть
        double ar_term = 0.0;
        for (int j = 0; j < p; ++j) {
            ar_term += ar_params[j] * series[t - j - 1];
        }
                
        // MA часть
        double ma_term = 0.0;
        for (int j = 0; j < q; ++j) {
            ma_term += ma_params[j] * noise[t - j - 1];
        }
                
        series[t] = ar_term + ma_term + noise[t];
    }
            
    // Возвращаем временной ряд с интегрированием
    for (int i = 0; i < d; ++i) {
        series = cumsum(series); // Интеграция путем кумуляции
    }
            
    return series;
}
    
        // Генерация временного ряда на основе модели SARIMAX.
        //
        // :param n: Длина временного ряда
        // :param ar_params: Коэффициенты авторегрессии (список)
        // :param d: Порядок дифференцирования
        // :param ma_params: Коэффициенты скользящего среднего (список)
        // :param seasonal_order: Параметры сезонности (p, d, q, s)
        // :param noise_std: Стандартное отклонение шума
        // :return: Временной ряд (numpy массив)
std::vector<double> TSGenerator::generate_sarimax_series(int n, const std::vector<double>& ar_params, int d, const std::vector<double>& ma_params, const std::vector<int>& seasonal_order, double noise_std = 1) {
    int p = static_cast<int>(ar_params.size());
    int q = static_cast<int>(ma_params.size());
    int s = seasonal_order[3];  // Длина сезонного цикла
            
    // Генерация нормального шума
    std::vector<double> noise(n, 0.0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, noise_std);
    for (int i = 0; i < n; ++i) {
        noise[i] = dist(gen);
    }
            
    // Генерация временного ряда
    std::vector<double> series(n, 0.0);
            
    // Применяем разности (различие)
    for (int i = 0; i < d; ++i) {
        series = diff(series, noise[0]);
    }
            
    int start_index = std::max({ p, q, seasonal_order[0] + seasonal_order[2] });
    for (int t = start_index; t < n; ++t) {
        // AR часть
        double ar_term = 0.0;
        for (int j = 0; j < p; ++j) {
            ar_term += ar_params[j] * series[t - j - 1];
        }
        // MA часть
        double ma_term = 0.0;
        for (int j = 0; j < q; ++j) {
            ma_term += ma_params[j] * noise[t - j - 1];
        }
                
        // Сезонные члены
        double seasonal_ar = 0.0;
        for (int j = 0; j < seasonal_order[0]; ++j) {
            seasonal_ar += seasonal_order[0] * series[t - s - j];
        }
        double seasonal_ma = 0.0;
        for (int j = 0; j < seasonal_order[2]; ++j) {
            seasonal_ma += seasonal_order[2] * noise[t - s - j];
        }
                
        // Генерация следующего значения
        series[t] = ar_term + ma_term + seasonal_ar + seasonal_ma + noise[t];
    }
            
    return series;
}
    