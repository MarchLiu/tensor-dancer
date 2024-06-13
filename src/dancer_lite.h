//
// Created by 刘鑫 on 2024/6/10.
//

#ifndef TENSOR_DANCER_MATRIX_LITE_H
#define TENSOR_DANCER_DANCER_LITE_H

#include <cstdio>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef USE_PG
#define dalloc(size) palloc(size)
#else
#define dalloc(size) malloc(size)
#endif

enum TOP_K {
    TOP_K_MIN = 0,
    TOP_K_MAX = 1,
};

typedef struct {
    size_t index;
    double value;
} IndexedDouble;

typedef struct {
    int index;
    float value;
} IndexedFloat;

int compare_ints(const void *a, const void *b);

int compare_double(const void *a, const void *b);

int compare_double_reverse(const void *a, const void *b);

int compare_indexed_double(const void *a, const void *b);

int compare_indexed_double_reverse(const void *a, const void *b);

int compare_indexed_double(const void *a, const void *b);
int compare_indexed_reverse_double(const void *a, const void *b);

double mean_double(const double *matrix, size_t len);
double sum_double(const double *matrix, size_t start, size_t stop);

void covariance_double(const double *matrix, double *covariance, size_t rows, size_t cols);
double square_frobenius_norm_double(const double *matrix, size_t len);
double frobenius_norm_double(const double *matrix, size_t len);
void top_k_indexes_double(const double *input, size_t len, size_t *indexes, size_t k, enum TOP_K mode);

float mean_float(const float *matrix, size_t len);
float sum_float(const float *matrix, size_t start, size_t stop);

void covariance_float(const float *matrix, float *covariance, size_t rows, size_t cols);
float square_frobenius_norm_float(const float *matrix, size_t len);
float frobenius_norm_float(const float *matrix, size_t len);
void top_k_indexes_float(const float *input, size_t len, size_t *indexes, size_t k, enum TOP_K mode);


#ifdef __cplusplus
};
#endif

#endif //TENSOR_DANCER_MATRIX_LITE_H
