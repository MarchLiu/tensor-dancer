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


class mem_streambuf : public std::streambuf {
public:
    mem_streambuf(char* begin, char* end) {
        setg(begin, begin, end);
    }
};

int write_matrix(struct Matrix *matrix, void *buffer, size_t size) {
    char* buffer_begin = static_cast<char*>(buffer);
    char* buffer_end = buffer_begin + size;

    mem_streambuf sb(buffer_begin, buffer_end);
    std::istream input(&sb);

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

struct Matrix *InitMatrixF32(void) {
    auto *result = (struct Matrix *) dalloc(sizeof(struct Matrix));
    result->magic = 0;
    result->type = GGML_TYPE_F32;
    result->rows = 0;
    result->columns = 0;
    result->data = nullptr;
    return result;
}

struct Matrix *InitMatrix(enum ggml_type type) {
    auto *result = (struct Matrix *) dalloc(sizeof(struct Matrix));
    result->magic = 0;
    result->type = type;
    result->rows = 0;
    result->columns = 0;
    result->data = nullptr;
    return result;
}

struct Matrix *CreateMatrix(enum ggml_type type, size_t rows, size_t columns) {
    size_t buffer_size = (rows * columns) * ggml_type_size(type);
    auto *result = (struct Matrix *) dalloc(sizeof(struct Matrix));
    result->magic = 0;
    result->type = type;
    result->rows = 0;
    result->columns = 0;
    result->data = dalloc(buffer_size);
    memset(result->data, 0, buffer_size);
    return result;
}


void FreeMatrix(struct Matrix *matrix) {
    if (matrix->data != nullptr) {
        dfree(matrix->data);
        matrix->data = nullptr;
    }
    dfree(matrix);
}

#ifdef __cplusplus
};
#endif
