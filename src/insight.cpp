//
// Created by 刘鑫 on 2024/6/11.
//

#include <iostream>
#include <random>
#include "ggml.h"
#include "dancer.h"

#ifdef __cplusplus
extern "C" {
#endif

void print_c_matrix_double(const double *matrix, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int idx = r * cols + c;
            std::cout << matrix[idx] << " ";
        }
        std::cout << std::endl;
    }
}

void print_c_matrix_float(const float *matrix, int rows, int cols) {
    for (int c = 0; c < cols; c++) {
        for (int r = 0; r < rows; r++) {
            int idx = r * cols + c;
            std::cout << matrix[idx] << " ";
        }
        std::cout << std::endl;
    }
}

void insight_c_matrix_float(const float *matrix, const char *title, int rows, int cols) {
    printf("%s: [\n", title);
    for (int r = 0; r < rows; r++) {
        std::cout << "\t[";
        for (int c = 0; c < cols; c++) {
            int idx = r * cols + c;
            std::cout << matrix[idx] << ",\t";
        }
        std::cout << "]," << std::endl;
    }
    printf("]\n");
}

void rand_c_matrix(double *matrix, int len) {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> distribution(0, 256);
    auto dice = std::bind(distribution, generator);

    for (int idx = 0; idx < len; idx++) {
        matrix[idx] = dice();
    }
}

void fill_rand_f32(ggml_tensor *tensor, float range) {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<float> distribution(0, range);
    auto dice = std::bind(distribution, generator);
    for (int i = 0; i < tensor->ne[0]; i++) {
        for (int j = 0; j < tensor->ne[1]; j++) {
            for (int k = 0; k < tensor->ne[2]; k++) {
                for (int l = 0; l < tensor->ne[3]; l++) {
                    ggml_set_f32_nd(tensor, i, j, k, l, dice());
                }
            }
        }
    }
}

void insight_matrix_tensor_f32(ggml_tensor *tensor, const char *title) {
    printf("%s(%lld, %lld):\n[\n", title, tensor->ne[0], tensor->ne[1]);
    for (int r = 0; r < tensor->ne[0]; r++) {
        std::cout << "\t[";
        for (int c = 0; c < tensor->ne[1]; c++) {
            std::cout << ggml_get_f32_nd(tensor, r, c, 0, 0) << ",\t";
        }
        std::cout << "]," << std::endl;
    }
    printf("]\n");
}

//void print_tensor(ggml_tensor *tensor) {
//    printf("%s(%lld, %lld, %lld, %lld):\n",
//           tensor->name, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
//    int dims = ggml_n_dims(tensor);
//
//    int ne_size = sizeof(int64_t)*GGML_MAX_DIMS;
//    int64_t* ne = (int64_t*)dalloc(ne_size);
//    memcpy(ne, tensor->ne, ne_size);
//
//    std::string base = "\t";
//    for (int i = 0; i < dims; ++i) {
//        std::string prompt(i, '\t');
//
//    }
//    for (int i = 0; i < tensor->ne[0]; i++) {
//        printf("\t[\n");
//        for (int j = 0; j < tensor->ne[1]; j++) {
//            printf("\t\t[");
//            for (int k = 0; k < tensor->ne[2]; k++) {
//                for (int l = 0; l < tensor->ne[3]; l++) {
//                    std::cout << ggml_get_f32_nd(tensor, i, j, k, l) << ", ";
//                }
//            }
//            printf("\t\t],\n");
//        }
//        printf("\t],\n");
//    }
//}

#ifdef __cplusplus
};
#endif