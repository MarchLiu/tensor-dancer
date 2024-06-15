//
// Created by 刘鑫 on 2024/6/11.
//

#include <assert.h>
#include <stdlib.h>
#include "cblas.h"
#include "src/dancer.h"

void *readall(char *filename) {
    FILE *file;
    long file_size;
    char *file_contents;

    // 打开文件
    file = fopen(filename, "rb"); // 以二进制读取模式打开文件

    // 移动文件指针到文件末尾，获取文件大小
    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    rewind(file); // 重置文件指针到文件开始位置

    // 为文件内容分配内存
    file_contents = (char *) dalloc(file_size + 1); // +1 为字符串的结束符'\0'
    if (file_contents == NULL) {
        fclose(file);
        perror("Error allocating memory");
    }

    // 读取文件内容
    size_t result = fread(file_contents, 1, file_size, file);
    if (result != file_size) {
        free(file_contents);
        fclose(file);
        perror("Error reading file");
    }

    // 添加字符串结束符
    file_contents[file_size] = '\0';

    // 关闭文件
    fclose(file);

    return file_contents;
}

void print_matrix(struct Matrix *matrix, char *title) {
    printf("%s:\n", title);
    printf("\t magic code: %u\n", matrix->magic);
    printf("\t type code: %u\n", matrix->type);
    printf("\t row: %zu\n", matrix->rows);
    printf("\t columns: %zu\n", matrix->columns);

//    float *pos = (float *) matrix->data;
//    for (size_t i = 0; i < matrix->rows; i++) {
//        for (size_t j = 0; j < matrix->columns; j++) {
//            printf("%s[%zu, %zu] = %f\n", title, i, j, pos[i * matrix->columns + j]);
//        }
//    }
}

int main(int argc, char **argv) {
    // assert(argc == 4);
    char *matrix_filename = argv[1];
    char *vector_filename = argv[2];
    void *m_content = readall(matrix_filename);
    struct Matrix *matrix = InitMatrixF32();
    write_matrix(matrix, m_content, 4194328);
    // load_matrix(matrix, matrix_filename);
    print_matrix(matrix, "matrix");

    struct Matrix *vector = dalloc(sizeof(struct Matrix));
    vector->rows = 1;
    vector->columns = matrix->columns;
    load_matrix(vector, vector_filename);
    // print_matrix(vector, "vector");

    struct Matrix *result = dalloc(sizeof(struct Matrix));
    result->rows = 1;
    result->columns = matrix->rows;
    result->data = dalloc(matrix->rows * sizeof(float));

    mul_matrix_vector_f32(matrix, vector->data, result->data);

    print_matrix(result, "result");

    dfree(result);
    dfree(vector);
    dfree(matrix);
    return EXIT_SUCCESS;
}
