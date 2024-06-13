//
// Created by 刘鑫 on 2024/6/13.
//

#include "matrix_lite.h"
#include "cblas.h"
#include <stddef.h>
#include <string.h>

int load_matrix(struct Matrix *matrix, void *buffer) {
    void *pos = buffer;
    // skip magic code check
    matrix->magic = 0;
    pos += sizeof(unsigned int);
    matrix->type = *(enum ggml_type *) pos;
    pos += sizeof(enum ggml_type);
    matrix->rows = *((size_t *) pos);
    pos += sizeof(size_t);
    matrix->columns = *((size_t *) pos);
    pos += sizeof(size_t);

    size_t items = matrix->rows * matrix->columns;
    size_t item_size = ggml_type_size(matrix->type);
    matrix->data = dalloc(items * item_size);
    memcpy(matrix->data, pos, items * item_size);
    return 0;
}

int mul_matrix_vector_f32(struct Matrix *matrix, float *vector, float *result) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int) matrix->rows, 1, (int) matrix->columns,
                1.0, (float *) matrix->data, (int) matrix->columns, vector, 1,
                0.0, result, 1);
    return 0;
}
