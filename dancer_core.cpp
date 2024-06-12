//
// Created by 刘鑫 on 2024/6/10.
//

#include <fstream>
#include "dancer_core.h"

#ifdef __cplusplus
extern "C" {
#endif

int fill_matrix(std::istream &input, Matrix &matrix) {

    input.read(reinterpret_cast<char *>(&matrix.magic), sizeof(unsigned int));
    input.read(reinterpret_cast<char *>(&matrix.type), sizeof(ggml_type));
    input.read(reinterpret_cast<char *>(&matrix.rows), sizeof(size_t));
    input.read(reinterpret_cast<char *>(&matrix.columns), sizeof(size_t));

    size_t size = matrix.rows * matrix.columns;
    size_t type_size = ggml_type_size(matrix.type);
    char *body = (char *) dalloc(size * type_size);
    char *pos = body;
    for (size_t i = 0; i < size; i++) {
        input.read(pos, static_cast<std::streamsize>(type_size));
        pos += type_size;
    }

    matrix.data = body;

    return 0;
}

int load_matrix(const char *filename, Matrix &matrix) {
    auto fin = std::ifstream(filename, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, filename);
        return -1;
    }
    auto status = fill_matrix(fin, matrix);
    fin.close();
    return status;
}

#ifdef __cplusplus
};
#endif

Matrix::Matrix() {
    this->type = GGML_TYPE_F32;
    this->rows = 0;
    this->columns = 0;
    this->data = nullptr;
}

Matrix::~Matrix() {
    if (this->data != nullptr) {
        dfree(data);
    }
}
