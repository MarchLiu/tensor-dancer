//
// Created by 刘鑫 on 2024/6/15.
//
#include "dancer.h"
#include <sstream>
#include <fstream>
#include <cblas.h>
#include "dancer_core.h"

#ifdef __cplusplus
extern "C" {
#endif

int write_matrix(struct Matrix *matrix, void *buffer, size_t size) {
    std::istringstream input((char *) buffer, size);
    return fill_matrix(*matrix, input);
}

int load_matrix(struct Matrix *matrix, const char *filename) {
    auto fin = std::ifstream(filename, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, filename);
        return -1;
    }
    auto status = fill_matrix(*matrix, fin);
    fin.close();
    return status;
}

int mul_matrix_vector_f32(struct Matrix *matrix, float *vector, float *result) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int) matrix->rows, 1, (int) matrix->columns,
                1.0, (float *) matrix->data, (int) matrix->columns, (float *) vector, 1,
                0.0, result, 1);
    return 0;
}

struct Matrix *InitMatrixF32() {
    struct Matrix *result = (struct Matrix *) malloc(sizeof(struct Matrix));
    result->magic = 0;
    result->type = GGML_TYPE_F32;
    result->rows = 0;
    result->columns = 0;
    result->data = nullptr;
    return result;
}

void FreeMatrix(struct Matrix* matrix) {
    if(matrix->data != nullptr){
        dfree(matrix->data);
        matrix->data = nullptr;
    }
    dfree(matrix);
}

#ifdef __cplusplus
};
#endif
