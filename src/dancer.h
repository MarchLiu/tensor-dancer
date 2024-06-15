//
// Created by 刘鑫 on 2024/6/15.
//

#ifndef TENSOR_DANCER_DANCER_H
#define TENSOR_DANCER_DANCER_H

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef USE_PG
#include "postgres.h"
#define dalloc(size) palloc(size);
#define dfree(data) pfree(data);
#else
#define dalloc(size) malloc(size);
#define dfree(data) free(data);
#endif

extern struct Matrix {
    unsigned int magic;
    enum ggml_type type;
    size_t rows;
    size_t columns;
    void *data;
} Matrix;

int load_matrix(struct Matrix *matrix, const char *filename);

int write_matrix(struct Matrix *matrix, void * buffer, size_t size);

int mul_matrix_vector_f32(struct Matrix *matrix, float * vector, float * result);

struct Matrix* InitMatrixF32(void);

struct Matrix *InitMatrix(enum ggml_type type);

struct Matrix *CreateMatrix(enum ggml_type type, size_t rows, size_t columns);

void FreeMatrix(struct Matrix* matrix);

#ifdef __cplusplus
};
#endif

#endif //TENSOR_DANCER_DANCER_H
