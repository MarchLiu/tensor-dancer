//
// Created by 刘鑫 on 2024/6/11.
//

#include <iostream>
#include <random>

#ifdef __cplusplus
extern "C" {
#endif

void print_c_matrix_double(const double *matrix, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int idx = r * cols + c;
            std::cout << matrix[idx] << " ";
        }
        std::cout << std::endl;
    }
}

void print_c_matrix_float(const float *matrix, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int idx = r * cols + c;
            std::cout << matrix[idx] << " ";
        }
        std::cout << std::endl;
    }
}

void rand_c_matrix(double *matrix, int len) {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> distribution(0, 256);
    auto dice = std::bind(distribution, generator);

    for (int idx = 0; idx < len; idx++) {
        matrix[idx] = dice();
    }
}
#ifdef __cplusplus
};
#endif