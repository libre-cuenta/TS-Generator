#include "headerGenerator.hpp"
#include <iostream>

int main() {
    // Example usage of TSGenerator functions
    int n = 100;
    std::vector<double> ar_params = {0.5, -0.3};
    std::vector<double> ma_params = {0.4};
    int d = 1;
    std::vector<int> seasonal_order = {1, 0, 1, 12}; // Example values: (p, d, q, s)

    std::vector<double> ar_series = TSGenerator::generate_ar_series(n, ar_params, 1);
    std::vector<double> ma_series = TSGenerator::generate_ma_series(n, ma_params, 1);
    std::vector<double> arma_series = TSGenerator::generate_arma_series(n, ar_params, ma_params, 1);
    std::vector<double> arima_series = TSGenerator::generate_arima_series(n, ar_params, d, ma_params, 1);
    std::vector<double> sarimax_series = TSGenerator::generate_sarimax_series(n, ar_params, d, ma_params, seasonal_order, 1);

    std::cout << "AR series first 5 values:" << std::endl;
    for (int i = 0; i < 5 && i < static_cast<int>(ar_series.size()); ++i) {
        std::cout << ar_series[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
