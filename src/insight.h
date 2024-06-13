//
// Created by 刘鑫 on 2024/6/11.
//

#ifndef TENSOR_DANCER_INSIGHT_H
#define TENSOR_DANCER_INSIGHT_H

#ifdef __cplusplus
extern "C" {
#endif

void print_c_matrix_double(const double *matrix, int rows, int cols);
void print_c_matrix_float(const float *matrix, int rows, int cols);

void rand_c_matrix(double *matrix, int len);

#ifdef __cplusplus
};
#endif

template<typename T>
void print_c_matrix(const T *matrix, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int idx = r * cols + c;
            std::cout << matrix[idx] << " ";
        }
        std::cout << std::endl;
    }
};

#endif //TENSOR_DANCER_INSIGHT_H
