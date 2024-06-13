//
// Created by 刘鑫 on 2024/6/13.
//

#ifndef TENSOR_DANCER_MATRIX_LITE_H
#define TENSOR_DANCER_MATRIX_LITE_H

#include "ggml.h"
#include "matrix_lite.h"

#ifdef USE_PG
#include "postgres.h"
#define dalloc(size) palloc(size);
#define dfree(data) pfree(data);
#else
#include <stdlib.h>
#define dalloc(size) malloc(size);
#define dfree(data) free(data);
#endif

struct Matrix {
    unsigned int magic;
    enum ggml_type type;
    size_t rows;
    size_t columns;
    void *data;
};


int mul_matrix_vector_f32(struct Matrix *matrix, float *vector, float *result);

int load_matrix(struct Matrix *matrix, void *buffer);

#endif //TENSOR_DANCER_MATRIX_LITE_H
