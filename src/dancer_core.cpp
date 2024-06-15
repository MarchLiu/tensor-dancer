//
// Created by 刘鑫 on 2024/6/10.
//

#include <fstream>
#include "cblas.h"
#include "dancer_core.h"
#include "dancer.h"
#include <sstream>

int fill_matrix(struct Matrix &matrix, std::istream &input) {

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