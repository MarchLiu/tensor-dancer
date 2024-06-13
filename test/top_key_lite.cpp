//
// Created by 刘鑫 on 2024/6/11.
//
#include <cassert>
#include <random>
#include <iostream>
#include "src/insight.h"
#include "src/dancer_lite.h"

int main(int argc, char **argv) {
    const size_t len = 32;
    const size_t rows = 2;
    const size_t cols = 16;
    const int k = 8;
    assert(rows * cols == len);

    auto *matrix = new double(len);
    auto *top = new double(k);
    auto *indexes = (size_t *) malloc(k * sizeof(int));

    std::cout << "init matrix " << rows << "*" << cols << std::endl;

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0, 256);
    auto dice = std::bind(distribution, generator);
    for (int idx = 0; idx < len; idx++) {
        matrix[idx] = dice();
    }

    print_c_matrix_double(matrix, rows, cols);

    top_k_indexes_double(matrix, len, indexes, k, TOP_K_MAX);

    for (int i = 0; i < k; i++) {
        top[i] = matrix[indexes[i]];
    }
    std::cout << "top max " << k << " items is " << std::endl;
    print_c_matrix_double(top, 1, 8);
    //clean
    delete matrix;
    delete top;
    free(indexes);

    return 0;
}