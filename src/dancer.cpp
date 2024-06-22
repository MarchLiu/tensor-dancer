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
    mem_streambuf(char *begin, char *end) {
        setg(begin, begin, end);
    }
};

int write_matrix(struct Matrix *matrix, void *buffer, size_t size) {
    char *buffer_begin = static_cast<char *>(buffer);
    char *buffer_end = buffer_begin + size;

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

void dancer_set_tensor_nd(ggml_tensor *tensor, int64_t i0, int64_t i1, int64_t i2, int64_t i3, void *value) {
    void *data =
            (char *) tensor->data + i0 * tensor->nb[0] + i1 * tensor->nb[1] + i2 * tensor->nb[2] + i3 * tensor->nb[3];
    memcpy(data, value, ggml_type_size(tensor->type));
}

struct ggml_tensor *load_matrix_as_tensor(struct ggml_context *ctx, const char *filename) {
    auto fin = std::ifstream(filename, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, filename);
        return nullptr;
    }
    unsigned int magic;
    ggml_type type;
    size_t rows;
    size_t columns;

    fin.read(reinterpret_cast<char *>(&magic), sizeof(unsigned int));
    fin.read(reinterpret_cast<char *>(&type), sizeof(ggml_type));
    fin.read(reinterpret_cast<char *>(&rows), sizeof(size_t));
    fin.read(reinterpret_cast<char *>(&columns), sizeof(size_t));

    //TODO ASSERT FILE FORMAT

    auto type_size = static_cast<std::streamsize>(ggml_type_size(type));
    char *buffer = new char[type_size];

    int64_t ne[2] = {static_cast<int64_t>(rows), static_cast<int64_t>(columns)};
    ggml_tensor *result = ggml_new_tensor(ctx, type, 2, ne);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < columns; ++j) {
            fin.read(buffer, type_size);
            dancer_set_tensor_nd(result,
                                 static_cast<int64_t>(i),
                                 static_cast<int64_t>(j),
                                 0, 0,
                                 buffer);
        }
    }
    return result;
}

struct MatrixHeader load_matrix_file_header(FILE *file) {
    struct MatrixHeader header{};
    fread(&header, sizeof(struct MatrixHeader), 1, file);
    return header;
}

struct MatrixHeader *matrix_header(const struct Matrix *matrix) {
    auto header = (struct MatrixHeader *) dalloc(sizeof(struct MatrixHeader));
    memcpy(header, matrix, sizeof(struct MatrixHeader));
    return header;
}

size_t save_matrix(const struct Matrix *matrix, const char *filename) {
    size_t count = 0;
    FILE *file = fopen(filename, "w+");
    auto header = (struct MatrixHeader *) matrix;
    count += write_matrix_header(header, file);
    size_t size = (matrix->rows) * (matrix->columns) * sizeof(ggml_type_size(matrix->type));
    count += fwrite(&(matrix->data), sizeof(size_t), size, file);
    fclose(file);
    return count;
}

size_t write_matrix_header(const struct MatrixHeader *header, FILE *file) {
    size_t count = 0;
    count += fwrite(&(header->magic), sizeof(unsigned int), 1, file);
    count += fwrite(&(header->type), sizeof(ggml_type), 1, file);
    count += fwrite(&(header->rows), sizeof(size_t), 1, file);
    count += fwrite(&(header->columns), sizeof(size_t), 1, file);
    return count;
}

size_t save_tensor_as_matrix(const struct ggml_tensor *tensor, const char *filename) {
    struct MatrixHeader header{
    };
    header.magic = MATRIX_MAGIC;
    header.type = tensor->type;
    header.rows = tensor->ne[0];
    header.columns = tensor->ne[1];
    FILE *file = fopen(filename, "w+");
    size_t size = sizeof(header);
    fwrite(&header, size, 1, file);
    for (size_t i = 0; i < header.rows; ++i) {
        for (size_t j = 0; j < header.columns; ++j) {
            float value = ggml_get_f32_nd(tensor, (int) i, (int) j, 0, 0);
            size += fwrite(&value, sizeof(float), 1, file);
        }
    }
    fflush(file);
    fclose(file);
    return size;
}

#ifdef __cplusplus
};
#endif
