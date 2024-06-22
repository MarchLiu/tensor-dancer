//
// Created by 刘鑫 on 2024/6/20.
//

#include <cstdio>
#include <iostream>
#include "dancer.h"
#include "insight.h"

int main(int argc, char **argv) {
    char *filename = argv[1];
    auto matrix = new struct Matrix();

    load_matrix(matrix, filename);

    char *magic = (char *) (&matrix->magic);

    std::cout << "Magic Code: " << magic << std::endl;
    std::cout << "Type Code: " << matrix->type << std::endl;
    std::cout << "Rows: " << matrix->rows << std::endl;
    std::cout << "Columns: " << matrix->columns << std::endl;

//    for(size_t i=0; i<matrix->rows; i++){
//        for(size_t j=0; j<matrix->columns; j++){
//            float value = ((float *)matrix->data)[i*matrix->columns +j];
//            printf("\t[%zu, %zu]=%f\n", i, j, value);
//        }
//    }
    insight_c_matrix_float((const float *)matrix->data, "Matrix",
                           matrix->rows, matrix->columns);

    delete matrix;
    return 0;
}