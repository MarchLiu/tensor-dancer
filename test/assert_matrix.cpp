//
// Created by 刘鑫 on 2024/6/11.
//

#include <cassert>
#include <iostream>
#include "src/dancer.h"
#include "cblas.h"

int main(int argc, char **argv) {
    assert(argc == 4);
    char *matrix_filename = argv[1];
    char *vector_filename = argv[2];
    char *expect_filename = argv[3];

    struct Matrix *matrix = InitMatrixF32();
    struct Matrix *vector = InitMatrixF32();
    struct Matrix *expect = InitMatrixF32();
    auto check = load_matrix(matrix, matrix_filename);
    check += load_matrix(vector, vector_filename);
    check += load_matrix(expect, expect_filename);
    if (check == 0) {
        std::cout << "matrix loaded" << std::endl;
        std::cout << "magic code:\t" << matrix->magic << std::endl;
        std::cout << "type code:\t" << matrix->type << std::endl;
        std::cout << "rows:\t" << matrix->rows << std::endl;
        std::cout << "columns:\t" << matrix->columns << std::endl;
        std::cout << "vector loaded" << std::endl;
        std::cout << "rows:\t" << vector->rows << std::endl;
        std::cout << "columns:\t" << vector->columns << std::endl;
        std::cout << "except loaded" << std::endl;
        std::cout << "rows:\t" << expect->rows << std::endl;
        std::cout << "columns:\t" << expect->columns << std::endl;

        auto *result = (float *) malloc(256 * sizeof(float));

//        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
//                    (int) matrix.rows, 1, (int) matrix.columns,
//                    1.0, (float *) matrix.data, (int) matrix.columns, (float *) vector.data, 1,
//                    0.0, result, 1);
        mul_matrix_vector_f32(matrix, (float *) vector->data, result);

        for (size_t i = 0; i < expect->columns; i++) {
            printf("element %zu is %f expect %f \n",
                   i, result[i], ((float *) expect->data)[i]);
        }

        free(result);
    } else {
        std::cerr << "matrix load failedm, error code: " << check << std::endl;
    }
}