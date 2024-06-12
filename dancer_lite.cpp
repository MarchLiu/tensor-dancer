//
// Created by 刘鑫 on 2024/6/10.
//

#include "dancer_lite.h"
#include <cmath>
#include <cstdlib>
#include <cassert>

#ifdef __cplusplus
extern "C" {
#endif

int compare_ints(const void *a, const void *b) {
    int arg1 = *(const int *) a;
    int arg2 = *(const int *) b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

int compare_double(const void *a, const void *b) {
    double arg1 = *(const double *) a;
    double arg2 = *(const double *) b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

int compare_double_reverse(const void *a, const void *b) {
    double arg1 = *(const double *) a;
    double arg2 = *(const double *) b;
    if (arg1 > arg2) return -1;
    if (arg1 < arg2) return 1;
    return 0;
}

int compare_indexed_double(const void *a, const void *b) {
    IndexedDouble arg1 = *(const IndexedDouble *) a;
    IndexedDouble arg2 = *(const IndexedDouble *) b;
    if (arg1.value < arg2.value) return -1;
    if (arg1.value > arg2.value) return 1;
    return 0;
}

int compare_indexed_double_reverse(const void *a, const void *b) {
    IndexedDouble arg1 = *(const IndexedDouble *) a;
    IndexedDouble arg2 = *(const IndexedDouble *) b;
    if (arg1.value > arg2.value) return -1;
    if (arg1.value < arg2.value) return 1;
    return 0;
}

double mean_double(const double *matrix, size_t len) {
    double result = 0.0;
    for (int i = 0; i < len; i++) {
        result += matrix[i];
    }
    result /= len;
    return result;
}

double sum_double(const double *matrix, size_t start, size_t stop) {
    double result = 0.0;
    for (int i = start; i < stop; i++) {
        result += matrix[i];
    }
    return result;
}

void covariance_double(const double *matrix, double *covariance, size_t rows, size_t cols) {
    // Calculate the outer product sum for the covariance matrix
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int k = j; k < cols; ++k) {
                covariance[j * cols + k] += matrix[i * cols + j] * matrix[i * cols + k];
            }
        }
    }

    // Normalize the covariance matrix
    double norm_factor = 1.0 / (rows - 1);
    for (int i = 0; i < cols * cols; ++i) {
        covariance[i] *= norm_factor;
    }


}

double square_frobenius_norm_double(const double *matrix, size_t len) {
    double accumulate = 0.0;
    for (int i = 0; i < len; ++i) {
        accumulate += matrix[i] * matrix[i];
    }
    return accumulate;
}

double frobenius_norm_double(const double *matrix, size_t len) {
    double accumulate = square_frobenius_norm_double(matrix, len);
    return sqrt(accumulate);
}

int max_index_double(const double *input, size_t start, size_t to) {
    assert(start < to);
    double value = input[start];
    int result = start;
    for (int i = start + 1; i < to; i++) {
        if (input[i] > value) {
            result = i;
            value = input[i];
        }
    }
    return result;
}

int min_index_double(const double *input, size_t start, size_t to) {
    assert(start < to);
    double value = input[start];
    int result = start;
    for (int i = start + 1; i < to; i++) {
        if (input[i] < value) {
            result = i;
            value = input[i];
        }
    }
    return result;
}

void top_k_indexes_double(const double *input, size_t len, size_t *indexes, size_t k, enum TOP_K mode) {
    int (*cmp)(const void *, const void *);
    int (*cmp_double)(const void *, const void *);
    if (mode == TOP_K_MIN) {
        cmp = &compare_indexed_double;
        cmp_double = &compare_double;
    } else {
        cmp = &compare_indexed_double_reverse;
        cmp_double = &compare_double_reverse;
    }

    IndexedDouble *buffer = (IndexedDouble *) dalloc(k * sizeof(IndexedDouble));
    for (int i = 0; i < k; i++) {
        buffer[i].index = i;
        buffer[i].value = input[i];
    }

    qsort(buffer, k, sizeof(IndexedDouble), cmp);

    int pos = k - 1;
    for (int i = k; i < len; i++) {
        int check = cmp_double(&input[i], &(buffer[pos].value));
        if (check == -1) {
            buffer[pos].index = i;
            buffer[pos].value = input[i];
            qsort(buffer, k, sizeof(IndexedDouble), cmp);
        }
    }

    qsort(buffer, k, sizeof(IndexedDouble), compare_indexed_double);
    for (int i = 0; i < k; i++) {
        indexes[i] = buffer[i].index;
    }

    free(buffer);
}

int compare_float(const void *a, const void *b) {
    double arg1 = *(const float *) a;
    double arg2 = *(const float *) b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

int compare_float_reverse(const void *a, const void *b) {
    double arg1 = *(const float *) a;
    double arg2 = *(const float *) b;
    if (arg1 > arg2) return -1;
    if (arg1 < arg2) return 1;
    return 0;
}

int compare_indexed_float(const void *a, const void *b) {
    IndexedFloat arg1 = *(const IndexedFloat *) a;
    IndexedFloat arg2 = *(const IndexedFloat *) b;
    if (arg1.value < arg2.value) return -1;
    if (arg1.value > arg2.value) return 1;
    return 0;
}

int compare_indexed_float_reverse(const void *a, const void *b) {
    IndexedFloat arg1 = *(const IndexedFloat *) a;
    IndexedFloat arg2 = *(const IndexedFloat *) b;
    if (arg1.value > arg2.value) return -1;
    if (arg1.value < arg2.value) return 1;
    return 0;
}

float mean_float(const float *matrix, size_t len) {
    float result = 0.0;
    for (int i = 0; i < len; i++) {
        result += matrix[i];
    }
    result /= len;
    return result;
}

float sum_float(const float *matrix, size_t start, size_t stop) {
    double result = 0.0;
    for (int i = start; i < stop; i++) {
        result += matrix[i];
    }
    return result;
}

void covariance_float(const float *matrix, float *covariance, size_t rows, size_t cols) {
    // Calculate the outer product sum for the covariance matrix
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int k = j; k < cols; ++k) {
                covariance[j * cols + k] += matrix[i * cols + j] * matrix[i * cols + k];
            }
        }
    }

    // Normalize the covariance matrix
    double norm_factor = 1.0 / (rows - 1);
    for (int i = 0; i < cols * cols; ++i) {
        covariance[i] *= norm_factor;
    }


}

float square_frobenius_norm_float(const float *matrix, size_t len) {
    float accumulate = 0.0;
    for (int i = 0; i < len; ++i) {
        accumulate += matrix[i] * matrix[i];
    }
    return accumulate;
}

float frobenius_norm_float(const float *matrix, size_t len) {
    double accumulate = square_frobenius_norm_float(matrix, len);
    return sqrt(accumulate);
}

int max_index_float(const float *input, size_t start, size_t to) {
    assert(start < to);
    double value = input[start];
    int result = start;
    for (int i = start + 1; i < to; i++) {
        if (input[i] > value) {
            result = i;
            value = input[i];
        }
    }
    return result;
}

int min_index_float(const float *input, size_t start, size_t to) {
    assert(start < to);
    double value = input[start];
    int result = start;
    for (int i = start + 1; i < to; i++) {
        if (input[i] < value) {
            result = i;
            value = input[i];
        }
    }
    return result;
}

void top_k_indexes_float(const float *input, size_t len, size_t *indexes, size_t k, enum TOP_K mode) {
    int (*cmp)(const void *, const void *);
    int (*cmp_float)(const void *, const void *);
    if (mode == TOP_K_MIN) {
        cmp = &compare_indexed_float;
        cmp_float = &compare_float;
    } else {
        cmp = &compare_indexed_float_reverse;
        cmp_float = &compare_float_reverse;
    }

    IndexedFloat *buffer = (IndexedFloat *) dalloc(k * sizeof(IndexedFloat));
    for (int i = 0; i < k; i++) {
        buffer[i].index = i;
        buffer[i].value = input[i];
    }

    qsort(buffer, k, sizeof(IndexedFloat), cmp);

    int pos = k - 1;
    for (int i = k; i < len; i++) {
        int check = cmp_float(&input[i], &(buffer[pos].value));
        if (check == -1) {
            buffer[pos].index = i;
            buffer[pos].value = input[i];
            qsort(buffer, k, sizeof(IndexedFloat), cmp);
        }
    }

    qsort(buffer, k, sizeof(IndexedFloat), compare_indexed_double);
    for (int i = 0; i < k; i++) {
        indexes[i] = buffer[i].index;
    }

    free(buffer);
}

#ifdef __cplusplus
};
#endif