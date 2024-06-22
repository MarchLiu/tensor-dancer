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

// 0x540x440x4D0x58 => TDMX = Tensor Dancer Matrix
const unsigned int MATRIX_MAGIC = 1481458772;

extern struct Matrix {
    unsigned int magic;
    enum ggml_type type;
    size_t rows;
    size_t columns;
    void *data;
} Matrix;

extern struct MatrixHeader {
    unsigned int magic;
    enum ggml_type type;
    size_t rows;
    size_t columns;
} MatrixHeader;

int load_matrix(struct Matrix *matrix, const char *filename);

int write_matrix(struct Matrix *matrix, void *buffer, size_t size);

int mul_matrix_vector_f32(struct Matrix *matrix, float *vector, float *result);

struct Matrix *InitMatrixF32(void);

struct Matrix *InitMatrix(enum ggml_type type);

struct Matrix *CreateMatrix(enum ggml_type type, size_t rows, size_t columns);

void FreeMatrix(struct Matrix *matrix);

struct ggml_tensor *load_matrix_as_tensor(struct ggml_context *ctx, const char *filename);

struct MatrixHeader load_matrix_file_header(FILE *file);

size_t save_matrix(const struct Matrix *matrix, const char *filename);

size_t save_tensor_as_matrix(const struct ggml_tensor *tensor, const char *filename);

size_t write_matrix_header(const struct MatrixHeader *matrix, FILE* file);

struct MatrixHeader* matrix_header(const struct Matrix* matrix);

#ifdef __cplusplus
};
#endif

#endif //TENSOR_DANCER_DANCER_H
