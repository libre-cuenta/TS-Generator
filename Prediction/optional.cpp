#include <iostream>
#include "headerOptional.hpp"


int main() {
    // Example usage of the translated models.
    // Создание dummy временного ряда данных
    std::vector<double> data = {100, 102, 101, 105, 110, 108, 107, 111, 115, 117, 120, 119, 118};
    
    // Пример использования ARIMA_model
    ARIMA_model arima(data);
    arima.train_test_split_ts(0.2);
    arima.find_optimal_d(2);
    arima.find_best_arima_params(3, 3);
    arima.fit_model();
    std::vector<double> forecast_arima = arima.predict(3);
    std::cout << "ARIMA Forecast:" << std::endl;
    for (double val : forecast_arima) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    arima.plot_predictions(forecast_arima);
    arima.plot_diagnostics();
    arima.plot_residuals();
    std::map<std::string, double> metrics_arima = arima.evaluate_predictions(arima.test, forecast_arima);
    std::cout << "Evaluation Metrics for ARIMA:" << std::endl;
    for (const auto& m : metrics_arima) {
        std::cout << m.first << ": " << m.second << std::endl;
    }
    arima.suggest_model_improvements();
    
    // Пример использования SARIMA_model
    SARIMA_model sarima(data);
    sarima.train_test_split_ts(0.2);
    sarima.find_optimal_D(12, 2);
    sarima.find_best_sarima_params(12, 2, 2, 1, 1);
    sarima.fit_model();
    std::vector<double> forecast_sarima = sarima.predict(3);
    std::cout << "SARIMA Forecast:" << std::endl;
    for (double val : forecast_sarima) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    sarima.plot_predictions(forecast_sarima);
    sarima.plot_diagnostics();
    sarima.plot_residuals();
    std::map<std::string, double> metrics_sarima = sarima.evaluate_predictions(sarima.test, forecast_sarima);
    std::cout << "Evaluation Metrics for SARIMA:" << std::endl;
    for (const auto& m : metrics_sarima) {
        std::cout << m.first << ": " << m.second << std::endl;
    }
    sarima.suggest_model_improvements();
    


    //ARCH
    // Example usage of the ARCH_model class
    vector<double> sample_data = {1.2, -0.5, 0.3, 2.1, -1.0, 0.7, 1.5, -0.3, 0.0, 0.8};
    ARCH_model arch_model_instance(sample_data);

    // Split data into train and test sets
    arch_model_instance.train_test_split_ts(0.2);

    // Check stationarity
    bool is_stationary = arch_model_instance.check_stationarity();
    cout << "Is stationary: " << (is_stationary ? "True" : "False") << endl;

    // Fit ARCH model
    arch_model_instance.fit_arch_model();

    // Optimize hyperparameters
    arch_model_instance.optimize_arch_parameters(3, 3);

    // Get historical volatility
    vector<double> hist_vol = arch_model_instance.historical_volatility();

    // Predict volatility for 5 steps ahead
    vector<double> forecast_vol = arch_model_instance.predict(5);

    // Evaluate predictions (using dummy actual and predicted values)
    map<string, double> eval_result = arch_model_instance.evaluate_predictions(hist_vol, forecast_vol);
    cout << "Evaluation RMSE: " << eval_result["RMSE"] << endl;

    // Plot results
    arch_model_instance.plot_results(hist_vol, forecast_vol);


    return 0;
}
  
