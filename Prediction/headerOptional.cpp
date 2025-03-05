#include "headerOptional.hpp"

// Helper function to compute differences similar to np.diff
std::vector<double> diff(const std::vector<double>& series, int n = 1) {
    std::vector<double> result;
    if(series.size() <= n) return result;
    for (size_t i = n; i < series.size(); i++) {
        result.push_back(series[i] - series[i - n]);
    }
    return result;
}


// Dummy implementation of the adfuller test similar to statsmodels.tsa.stattools.adfuller
ADFResult adfuller(const std::vector<double>& series) {
    // Dummy implementation: For illustration, we assume the adfStatistic and pValue are computed.
    // In a real implementation, proper statistical tests would be applied.
    ADFResult result;
    result.adfStatistic = -3.5; // dummy value
    result.pValue = 0.04;       // dummy value: change to >0.05 if non-stationary is desired
    result.criticalValues["1%"] = -3.5;
    result.criticalValues["5%"] = -2.9;
    result.criticalValues["10%"] = -2.6;
    return result;
}

// Helper function to compute Mean Squared Error
double mean_squared_error(const std::vector<double>& actual, const std::vector<double>& predicted) {
    if(actual.size() != predicted.size() || actual.empty()) return 0.0;
    double mse = 0.0;
    for (size_t i = 0; i < actual.size(); i++) {
        mse += (actual[i] - predicted[i]) * (actual[i] - predicted[i]);
    }
    return mse / actual.size();
}

// Helper function to compute Mean Absolute Error
double mean_absolute_error(const std::vector<double>& actual, const std::vector<double>& predicted) {
    if(actual.size() != predicted.size() || actual.empty()) return 0.0;
    double mae = 0.0;
    for (size_t i = 0; i < actual.size(); i++) {
        mae += std::fabs(actual[i] - predicted[i]);
    }
    return mae / actual.size();
}

// Helper function to compute R2 Score
double r2_score(const std::vector<double>& actual, const std::vector<double>& predicted) {
    if(actual.size() != predicted.size() || actual.empty()) return 0.0;
    double mean_actual = 0.0;
    for (double val : actual) {
        mean_actual += val;
    }
    mean_actual /= actual.size();
    double ss_tot = 0.0, ss_res = 0.0;
    for (size_t i = 0; i < actual.size(); i++) {
        ss_res += (actual[i] - predicted[i]) * (actual[i] - predicted[i]);
        ss_tot += (actual[i] - mean_actual) * (actual[i] - mean_actual);
    }
    return 1 - (ss_res / ss_tot);
}

// Helper function to compute autocorrelation function (ACF) up to nlags
std::vector<double> acf(const std::vector<double>& series, int nlags = 40) {
    std::vector<double> acf_values;
    size_t n = series.size();
    if(n == 0) return acf_values;
    double mean = 0.0;
    for (double v : series) {
        mean += v;
    }
    mean /= n;
    double denom = 0.0;
    for (size_t i = 0; i < n; i++) {
        denom += (series[i] - mean) * (series[i] - mean);
    }
    for (int lag = 0; lag <= nlags; lag++) {
        double num = 0.0;
        for (size_t i = 0; i < n - lag; i++) {
            num += (series[i] - mean) * (series[i + lag] - mean);
        }
        acf_values.push_back((denom != 0) ? num / denom : 0.0);
    }
    return acf_values;
}

// Dummy plotting function to simulate matplotlib plotting
void show_plot(const std::string &plotDescription) {
    std::cout << plotDescription << " [Plot displayed]" << std::endl;
}



// Base abstract class Korezin_TS


// Dummy placeholder for model; in Python this holds ARIMA/SARIMAX model objects.
// Here we simulate by storing the training data and order for use in forecast.


// Dummy fit function that returns itself.
DummyModel& DummyModel::fit() {
    return *this;
}
        
// Dummy forecast function that predicts the last value repeatedly.
std::vector<double> DummyModel::forecast(int steps) {
    std::vector<double> forecast;
    if(trainData.empty()) return forecast;
    double last = trainData.back();
    for (int i = 0; i < steps; i++) {
        forecast.push_back(last);
    }
    return forecast;
}
        
// Dummy plot diagnostics function.
void DummyModel::plot_diagnostics(const std::tuple<int,int>& figsize) {
    std::ostringstream oss;
    oss << "Diagnostics plot with figsize(" << std::get<0>(figsize) << ", " << std::get<1>(figsize) << ")";
    show_plot(oss.str());
}

    
    // Constructor
Korezin_TS::Korezin_TS(const std::vector<double>& data_input) : data(data_input), aic(std::numeric_limits<double>::infinity()) {
    // Initialize order to (1, 1, 1)
    order = std::make_tuple(1, 1, 1);
}
    
    // Virtual destructor
Korezin_TS:: ~Korezin_TS() {}
    
    // Method: Разделение временного ряда на обучающую и тестовую выборки
Korezin_TS* Korezin_TS::train_test_split_ts(double test_size = 0.2) {
    // If data is not in vector<double>, nothing to convert in C++ since we assume it is.
    size_t train_size = static_cast<size_t>(data.size() * (1 - test_size));
    train.assign(data.begin(), data.begin() + train_size);
    test.assign(data.begin() + train_size, data.end());
    return this;
}
    
    // Method: Проверка стационарности временного ряда с помощью теста Дики-Фуллера
bool Korezin_TS::check_stationarity() {
    ADFResult result = adfuller(data);
    std::cout << "ADF Statistic: " << result.adfStatistic << std::endl;
    std::cout << "p-value: " << result.pValue << std::endl;
    std::cout << "Critical values:" << std::endl;
    for (const auto& kv : result.criticalValues) {
        std::cout << "\t" << kv.first << ": " << kv.second << std::endl;
    }
    // Если p-value > 0.05, ряд нестационарен
    return (result.pValue <= 0.05);
}
    
    // Method: Определение оптимального параметра d путем последовательного дифференцирования
    // Note: This replicates the original code exactly even if there is a logical issue.
Korezin_TS* Korezin_TS::find_optimal_d(int max_d = 2) {
    int d = 0;
    std::vector<double> series = data;
    
    while(d <= max_d) {
        if(check_stationarity()) {
            std::get<1>(order) = d;
            std::cout << "Optional d: " << d << std::endl;
            return this;
        }
        series = diff(series);
        d += 1;
    }
    std::get<1>(order) = d;
    std::cout << "Optional d: " << d << std::endl;
    return this;
}
    
    // Method: Построение прогноза на n_steps шагов вперед
std::vector<double> Korezin_TS::predict(int steps) {
    // In the dummy implementation, fit the model and then forecast
    std::vector<double> forecast = model.fit().forecast(steps);
    return forecast;
}
    
    // Method: Оценка качества прогноза
std::map<std::string, double> Korezin_TS::evaluate_predictions(const std::vector<double>& actual, const std::vector<double>& predicted) {
    double mse = mean_squared_error(actual, predicted);
    double rmse = std::sqrt(mse);
    double mae = mean_absolute_error(actual, predicted);
    double r2 = r2_score(actual, predicted);
        
    std::map<std::string, double> metrics;
    metrics["MSE"] = mse;
    metrics["RMSE"] = rmse;
    metrics["MAE"] = mae;
    metrics["R2"] = r2;
    return metrics;
}
    
    // Method: suggest_model_improvements
void Korezin_TS::suggest_model_improvements() {
    std::vector<double> acf_values = acf(residuals);
    
    if (acf_values.size() > 1 && std::fabs(acf_values[1]) > 0.2)
        std::cout << "Рекомендуется увеличить порядок AR" << std::endl;
    else
        std::cout << "Порядок AR соответствует модели" << std::endl;
    
    if (acf_values.size() > 0 && std::fabs(acf_values.back()) > 0.2)
        std::cout << "Рекомендуется увеличить порядок MA" << std::endl;
    else
        std::cout << "Порядок MA соответствует модели" << std::endl;
}

    
    // The following abstract method is commented out in the original code
    // virtual Korezin_TS* predict_by_step(int step, const std::string& predicted_view) = 0;
    
    // Method: Построение диагностических графиков
void Korezin_TS::plot_diagnostics() {
    // In the dummy implementation, we simulate a diagnostic plot.
    // figsize is set to (15, 12) as in the original code.
    model.fit().plot_diagnostics(std::make_tuple(15, 12));
    // In an actual implementation, plotting code would be here.
}
    
    // Pure virtual method: plot_predictions
void Korezin_TS::plot_predictions(const std::vector<double>& predictions) = 0;
    
    // Method: Анализ остатков модели
void Korezin_TS::plot_residuals() {
    // Simulate conversion of residuals into a DataFrame by just using the vector.
    // Simulate subplots by printing titles for each subplot.
    std::cout << "Plotting residuals:" << std::endl;
    // График остатков
    std::cout << "Остатки plot: ось X - Время, ось Y - Значение остатков" << std::endl;
    // Гистограмма остатков (KDE plot simulated)
    std::cout << "Плотность распределения остатков (KDE plot)" << std::endl;
    // Q-Q график
    std::cout << "Q-Q График (line=45)" << std::endl;
    // Автокорреляция остатков
    std::vector<double> acf_values = acf(residuals, 40);
    std::cout << "Автокорреляция остатков:" << std::endl;
    for (size_t i = 0; i < acf_values.size(); i++) {
        std::cout << "Lag " << i << ": " << acf_values[i] << std::endl;
    }
    std::cout << "Горизонтальные линии: y=0, y=-1.96/sqrt(n), y=1.96/sqrt(n)" << std::endl;
    // Simulate tight layout and show plot
    show_plot("Диагностические графики остатков");
}



    // Constructor: Инициализация модели SARIMA (note that the comment is same as original)
ARIMA_model::ARIMA_model(const std::vector<double>& data) : Korezin_TS(data) {
    // Order is already initialized in base class constructor
}
    
    // Method: Подбор оптимальных параметров p, d, q для модели ARIMA
ARIMA_model* ARIMA_model::find_best_arima_params(int max_p, int max_q) {
    double best_aic = std::numeric_limits<double>::infinity();
    std::tuple<int, int, int> best_params = order;
    // Перебираем все возможные комбинации параметров p, d и q
    for (int p = 0; p <= max_p; p++) {
        for (int q = 0; q <= max_q; q++) {
            if(p == 0 && q == 0) {
                continue;
            }
            try {
            // Симуляция создания и обучения модели ARIMA
                // Dummy AIC is computed as (p+q)*100 minus a small factor depending on train size.
                double current_aic = (p + q) * 100.0;
                std::cout << current_aic << std::endl;
                
                if(current_aic < best_aic) {
                    best_aic = current_aic;
                    best_params = std::make_tuple(p, std::get<1>(order), q);
                }
            } catch (std::exception& e) {
                continue;
            }
        }
    }
    order = best_params;
    aic = best_aic;
    return this;
}
    
    // Method: Обучение модели ARMA с заданными параметрами
ARIMA_model* ARIMA_model::fit_model() {
    // In the dummy implementation, we simulate model fitting.
    // Set the dummy model's training data and order.
    model.trainData = train;
    model.order = order;
    // Simulate residuals as differences between train data and their mean.
    double meanVal = 0.0;
    for (double v : train) meanVal += v;
    meanVal /= (train.empty() ? 1 : train.size());
    residuals.clear();
    for (double v : train) {
        residuals.push_back(v - meanVal);
    }
    return this;
}
    
    // Method: Визуализация результатов прогнозирования
void ARIMA_model::plot_predictions(const std::vector<double>& predictions) {
    std::string title = "ARIMA Прогноз";
    std::cout << "Plot Title: " << title << std::endl;
    
    // Построение обучающих данных
    std::cout << "Plotting Обучающие данные (color: #ff0000)" << std::endl;
    // Построение тестовых данных
    std::cout << "Plotting Тестовые данные (color: #00ff00)" << std::endl;
    // Построение прогноза
    std::cout << "Plotting Прогноз (color: #0000ff, linestyle: --)" << std::endl;
        
    std::cout << "Оси: X - Время, Y - Значение" << std::endl;
    std::cout << "Легенда и Сетка установлены" << std::endl;
    
    show_plot("ARIMA Прогноз график");
}


// Derived class SARIMA_model that inherits from Korezin_TS

SARIMA_model::SARIMA_model(const std::vector<double>& data) : Korezin_TS(data) {
    // Initialize seasonal_order to (1,1,1,12)
    seasonal_order = std::make_tuple(1, 1, 1, 12);
}
    
// Method: Определение оптимального сезонного параметра D
SARIMA_model* SARIMA_model::find_optimal_D(int season_length, int max_D = 2) {
    int D = 0;
    std::vector<double> series = data;
    while(D <= max_D) {
        if(check_stationarity()) {
            std::get<1>(seasonal_order) = D;
            std::cout << "Optional D: " << D << std::endl;
            return this;
        }
        series = diff(series, season_length);
        D += 1;
    }
    std::get<1>(seasonal_order) = D;
    return this;
}
    
// Method: Поиск оптимальных параметров SARIMA по сетке
SARIMA_model* SARIMA_model::find_best_sarima_params(int season_length,
                                      int max_p, int max_q,
                                      int max_P, int max_Q) {
    double best_aic = std::numeric_limits<double>::infinity();
    std::tuple<int, int, int> best_order = order;
    std::tuple<int, int, int, int> best_seasonal = seasonal_order;
        
    for (int p = 0; p <= max_p; p++) {
        for (int q = 0; q <= max_q; q++) {
            for (int P = 0; P <= max_P; P++) {
                for (int Q = 0; Q <= max_Q; Q++) {
                    // Формируем параметры модели
                    std::tuple<int,int,int> current_order = std::make_tuple(p, std::get<1>(order), q);
                    std::tuple<int,int,int,int> current_seasonal = std::make_tuple(P, std::get<1>(seasonal_order), Q, season_length);
                    try {
                        // Симуляция создания и обучения модели SARIMA
                        // Dummy AIC is computed as (p+q+P+Q)*80 (dummy computation)
                        double current_aic = (p + q + P + Q) * 80.0;
                        if(current_aic < best_aic) {
                            best_aic = current_aic;
                            best_order = current_order;
                            best_seasonal = current_seasonal;
                        }
                        std::cout << "SARIMA(" << p << "," << std::get<1>(order) << "," << q << ")x(" 
                                  << P << "," << std::get<1>(seasonal_order) << "," << Q << "," << season_length 
                                  << ") - AIC:" << current_aic << std::endl;
                    } catch (std::exception& e) {
                        continue;
                    }
                }
            }
        }
    }
    order = best_order;
    seasonal_order = best_seasonal;
    aic = best_aic;
    return this;
}
    
    // Method: Обучение модели SARIMA с заданными параметрами
SARIMA_model* SARIMA_model::fit_model() {
    // In the dummy implementation, we simulate model fitting for SARIMA.
    model.trainData = train;
    model.order = order;
    // For seasonal_order, assign it to the dummy model as well.
    model.seasonal_order = seasonal_order;
    double meanVal = 0.0;
    for (double v : train) meanVal += v;
    meanVal /= (train.empty() ? 1 : train.size());
    residuals.clear();
    for (double v : train) {
        residuals.push_back(v - meanVal);
    }
    return this;
}
    
    // Method: Визуализация результатов прогнозирования
void SARIMA_model::plot_predictions(const std::vector<double>& predictions) {
    std::string title = "SARIMA Прогноз";
    std::cout << "Plot Title: " << title << std::endl;
    
    // Построение обучающих данных
    std::cout << "Plotting Обучающие данные (color: #ff0000)" << std::endl;
    // Построение тестовых данных
    std::cout << "Plotting Тестовые данные (color: #00ff00)" << std::endl;
    // Построение прогноза
    std::cout << "Plotting Прогноз (color: #0000ff, linestyle: --)" << std::endl;
        
    std::cout << "Оси: X - Время, Y - Значение" << std::endl;
    std::cout << "Легенда и Сетка установлены" << std::endl;
    
    show_plot("SARIMA Прогноз график");
}



//ARCH
ARCH_model::ARCH_model(const vector<double>& data) {
    this->data = data;
    this->train = vector<double>();
    this->test = vector<double>();
    this->order = make_pair(1, 1);
    this->aic = 0.0;
    this->model = nullptr;
    this->residuals = vector<double>();
}

// Разделение временного ряда на обучающую и тестовую выборки
ARCH_model& ARCH_model::train_test_split_ts(double test_size = 0.2) {
    // In C++, we assume the data is already a vector<double>
    int train_size = static_cast<int>(data.size() * (1 - test_size));
    train = vector<double>(data.begin(), data.begin() + train_size);
    test = vector<double>(data.begin() + train_size, data.end());
    return *this;
}

// Проверка стационарности временного ряда с помощью теста Дики-Фуллера
bool ARCH_model::check_stationarity() {
    ADFResult result = adfuller(data);
    cout << "ADF Statistic: " << result.statistic << endl;
    cout << "p-value: " << result.pvalue << endl;
    cout << "Critical values:" << endl;
    for (const auto &kv : result.crit_values) {
        cout << "\t" << kv.first << ": " << kv.second << endl;
    }
    // Если p-value > 0.05, ряд нестационарен
    return (result.pvalue <= 0.05);
}

// Функция для построения и обучения модели ARCH
ARCH_model& ARCH_model::fit_arch_model() {
    // Create a new ArchModel instance with vol = "ARCH"
    if(model != nullptr) {
        delete model;
    }
    model = new ArchModel(data, "ARCH", order.first, order.second);
    return *this;
}

ARCH_model& ARCH_model::fit_garch_model() {
    // Create a new ArchModel instance with vol = "GARCH"
    if(model != nullptr) {
        delete model;
    }
    model = new ArchModel(data, "GARCH", order.first, order.second);
    return *this;
}

// Добавление кросс-валидации
pair<double, double> ARCH_model::cross_validate_model(const vector<double>& returns, int n_splits = 5) {
    vector<pair<vector<int>, vector<int>>> tscv = timeseries_split(returns, n_splits);
    vector<double> rmse_scores;
    
    for (size_t i = 0; i < tscv.size(); i++) {
        vector<int> train_index = tscv[i].first;
        vector<int> test_index = tscv[i].second;
        vector<double> train_returns, test_returns;
        for (int idx : train_index) {
            train_returns.push_back(returns[idx]);
        }
        for (int idx : test_index) {
            test_returns.push_back(returns[idx]);
        }
        
        // Fit model and forecast on test_returns horizon
        if(model == nullptr) {
            // If model hasn't been set, default to ARCH model.
            model = new ArchModel(data, "ARCH", order.first, order.second);
        }
        FitResult fit_result = model->fit();
        typename FitResult::ForecastResult forecast = fit_result.forecast(test_returns.size());
        // Get the last row of forecast variance and compute square root to get predicted volatility
        vector<double> predicted_vol;
        if (!forecast.variance.empty()) {
            vector<double> last_row = forecast.variance.back();
            for (double val : last_row) {
                predicted_vol.push_back(sqrt(val));
            }
        }
        // Actual volatility as absolute values of test_returns
        vector<double> actual_vol;
        for (double val : test_returns) {
            actual_vol.push_back(fabs(val));
        }
        // Ensure sizes match
        size_t len = min(actual_vol.size(), predicted_vol.size());
        vector<double> act(actual_vol.begin(), actual_vol.begin() + len);
        vector<double> pred(predicted_vol.begin(), predicted_vol.begin() + len);
        double rmse = sqrt(mean_squared_error(act, pred));
        rmse_scores.push_back(rmse);
    }
    
    // Compute mean and standard deviation of rmse_scores
    double sum = accumulate(rmse_scores.begin(), rmse_scores.end(), 0.0);
    double mean_rmse = rmse_scores.empty() ? 0.0 : sum / rmse_scores.size();
    double sq_sum = 0.0;
    for (double score : rmse_scores) {
        sq_sum += (score - mean_rmse) * (score - mean_rmse);
    }
    double std_rmse = rmse_scores.empty() ? 0.0 : sqrt(sq_sum / rmse_scores.size());
    return make_pair(mean_rmse, std_rmse);
}

// Добавление оптимизации гиперпараметров
ARCH_model& ARCH_model::optimize_arch_parameters(int max_p, int max_q) {
    double best_aic = numeric_limits<double>::infinity();
    pair<int, int> best_params = order;
    
    for (int p = 0; p <= max_p; p++) {
        for (int q = 0; q <= max_q; q++) {
            if (p == 0 && q == 0)
                continue;
            
            try {
                ArchModel model_temp(data, "Garch", p, q);
                // Simulate fitting the model with disp turned off.
                FitResult results = model_temp.fit();
                cout << results.aic << endl;
                
                if (results.aic < best_aic) {
                    best_aic = results.aic;
                    best_params = make_pair(p, q);
                }
            } catch (exception &e) {
                continue;
            }
        }
    }
    
    order = best_params;
    aic = best_aic;
    return *this;
}

vector<double> ARCH_model::historical_volatility() {
    if(model == nullptr) {
        // If model hasn't been set, default to ARCH model.
        model = new ArchModel(data, "ARCH", order.first, order.second);
    }
    FitResult fit_result = model->fit();
    vector<double> hist_vol;
    for (double val : fit_result.conditional_volatility) {
        hist_vol.push_back(sqrt(val));
    }
    return hist_vol;
}

// Функция для получения прогноза волатильности
vector<double> ARCH_model::predict(int steps) {
    if(model == nullptr) {
        // If model hasn't been set, default to ARCH model.
        model = new ArchModel(data, "ARCH", order.first, order.second);
    }
    FitResult fit_result = model->fit();
    typename FitResult::ForecastResult forecast = fit_result.forecast(steps);
    vector<double> predicted_vol;
    if (!forecast.variance.empty()) {
        vector<double> last_row = forecast.variance.back();
        for (double val : last_row) {
            predicted_vol.push_back(sqrt(val));
        }
    }
    return predicted_vol;
}

map<string, double> ARCH_model::evaluate_predictions(const vector<double>& actual, const vector<double>& predicted) {
    if(model == nullptr) {
        // If model hasn't been set, default to ARCH model.
        model = new ArchModel(data, "ARCH", order.first, order.second);
    }
    FitResult fit_result = model->fit();
    typename FitResult::ForecastResult forecast = fit_result.forecast(test.size());
    vector<double> predicted_vol;
    if (!forecast.variance.empty()) {
        vector<double> last_row = forecast.variance.back();
        for (double val : last_row) {
            predicted_vol.push_back(sqrt(val));
        }
    }
    // Actual volatility as absolute values of test data
    vector<double> actual_vol;
    for (double val : test) {
        actual_vol.push_back(fabs(val));
    }
    size_t len = min(actual_vol.size(), predicted_vol.size());
    vector<double> act(actual_vol.begin(), actual_vol.begin() + len);
    vector<double> pred(predicted_vol.begin(), predicted_vol.begin() + len);
    double rmse = sqrt(mean_squared_error(act, pred));
    map<string, double> result;
    result["RMSE"] = rmse;
    return result;
}

// Функция для визуализации результатов
void ARCH_model::plot_results(const vector<double>& volatility, const vector<double>& forecast_vals) {
    // Построение графика доходности
    plt::figure_size(1200, 600);
    
    // Построение графика возвратов (Returns)
    plt::subplot(2, 1, 1);
    plt::plot(data, {{"label", "Returns"}});
    plt::title("Returns and Volatility");
    plt::legend();
    
    // Построение графика волатильности
    plt::subplot(2, 1, 2);
    plt::plot(volatility, {{"label", "Historical Volatility"}});
    
    // Create x-values for forecast: range from volatility.size() to volatility.size() + forecast_vals.size()
    vector<double> forecast_x;
    for (size_t i = volatility.size(); i < volatility.size() + forecast_vals.size(); i++) {
        forecast_x.push_back(static_cast<double>(i));
    }
    plt::plot(forecast_x, forecast_vals, {{"label", "Forecasted Volatility"}});
    plt::legend();
    
    plt::tight_layout();
    plt::show();
}

// Destructor to free allocated resources
ARCH_model::~ARCH_model() {
    if(model != nullptr) {
        delete model;
    }
}


//ARIMAGARCHPredictor
ARIMAGARCHPredictor::ARIMAGARCHPredictor(ArimaModel arima_model, GarchModel garch_model) 
            : arima_model(arima_model), garch_model(garch_model) {}
    
void ARIMAGARCHPredictor::fit(ArimaModel arima_model, GarchModel garch_model) {
    this->arima_model = arima_model;
    this->garch_model = garch_model;
}
    
std::map<std::string, std::vector<double>> ARIMAGARCHPredictor::predict(int steps) {
    // Forecast ARIMA
    auto arima_forecast = arima_model.fit().forecast(steps);
            
    // Forecast GARCH volatility
    auto garch_forecast = garch_model.fit().forecast(steps);
    std::vector<double> volatility(garch_forecast.variance.end() - steps, garch_forecast.variance.end());
    std::transform(volatility.begin(), volatility.end(), volatility.begin(), [](double var) { return std::sqrt(var); });
    
    // Combined forecast
    std::map<std::string, std::vector<double>> forecast;
    forecast["mean"] = arima_forecast;
    forecast["volatility"] = volatility;
    forecast["lower_bound"] = arima_forecast; // Placeholder for lower bound calculation
    forecast["upper_bound"] = arima_forecast; // Placeholder for upper bound calculation
    
    for (size_t i = 0; i < arima_forecast.size(); ++i) {
        forecast["lower_bound"].push_back(arima_forecast[i] - 1.96 * volatility[i]);
        forecast["upper_bound"].push_back(arima_forecast[i] + 1.96 * volatility[i]);
    }
    
    return forecast;
}
    
std::map<std::string, double> ARIMAGARCHPredictor::evaluate_model(const std::vector<double>& predictions, const std::vector<double>& actual) {
    if (predictions.size() != actual.size()) {
        throw std::invalid_argument("Predictions and actual values must have the same length.");
    }
    
    double mse = std::inner_product(predictions.begin(), predictions.end(), actual.begin(), 0.0, std::plus<double>(), [](double pred, double act) {
        return std::pow(pred - act, 2);
    }) / predictions.size();
    double rmse = std::sqrt(mse);
    
    // Check coverage of confidence interval
    int in_interval = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] >= predictions[i] - 1.96 * predictions[i] && actual[i] <= predictions[i] + 1.96 * predictions[i]) {
            in_interval++;
        }
    }
    double interval_coverage = static_cast<double>(in_interval) / actual.size();
    
    return { {"rmse", rmse}, {"interval_coverage", interval_coverage} };
}
    
void ARIMAGARCHPredictor::plot_results(const std::vector<double>& actual, const std::map<std::string, std::vector<double>>& predictions, const std::string& title = "ARIMA+GARCH Forecast") {
    plt::figure_size(1200, 600);
    
    // Plot actual values
    plt::plot(actual, "b-", {{"label", "Actual"}});
    
    // Plot forecast
    plt::plot(predictions.at("mean"), "r-", {{"label", "Forecast"}});
    
    // Plot confidence interval
    plt::fill_between(predictions.at("lower_bound"), predictions.at("upper_bound"), "gray", {{"alpha", "0.2"}, {"label", "95% Confidence Interval"}});
    
    plt::title(title);
    plt::xlabel("Time");
    plt::ylabel("Value");
    plt::legend();
    plt::grid(true);
    plt::show();
}