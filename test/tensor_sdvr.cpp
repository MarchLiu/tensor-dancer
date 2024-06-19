//
// Created by 刘鑫 on 2024/6/17.
//
#include "ggml.h"
#include "dancer.h"
#include "ggml_dancer.h"
#include "cblas.h"
#include "lapacke.h"
#include "insight.h"



int main(int argc, char **argv) {
    struct ggml_init_params params = ggml_init_params();
    params.mem_size = 1024 * 1024 * 1204;
    ggml_context *ctx = ggml_init(params);

    const int64_t ne[2] = {5, 5};
    ggml_tensor *matrix = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);
    fill_rand_f32(matrix, 256.0);

    std::string title = "Matrix";
    insight_c_matrix_float((float *) matrix->data, title.c_str(), 5, 5);

    const int64_t sne[2] = {1, 5};
    ggml_tensor *tensor_S = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, sne);

    const int64_t vtne[2] = {5, 5};
    ggml_tensor *tensor_VT = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, vtne);

    lapack_svd_rf32(matrix, tensor_S, tensor_VT);

    std::string s_title = "Singulars";
    insight_c_matrix_float((float *) tensor_S->data, s_title.c_str(), 1, 5);

    std::string vt_title = "VT";
    insight_c_matrix_float((float *) tensor_VT->data, vt_title.c_str(), 5, 5);

    ggml_free(ctx);
    return 0;
}