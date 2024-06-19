//
// Created by 刘鑫 on 2024/6/18.
//
#include "ggml.h"
#include "insight.h"

const float esp = 1e-10;

ggml_tensor *tensor_covariance(ggml_context *ctx, ggml_tensor *sample_set) {
    // 求特征平均值
    ggml_tensor *mean = ggml_mean(ctx, sample_set);

    struct ggml_tensor *center = ggml_repeat(ctx, mean, sample_set);

    // 中心化
    ggml_tensor *base_line = ggml_sub(ctx, sample_set, center);

    // 求无偏方差
    ggml_tensor *sqr = ggml_sqr(ctx, base_line);
    ggml_tensor *sum_rows = ggml_sum_rows(ctx, sqr);
    auto dnm = 1.0f / ((float) sample_set->ne[0] - 1);
    ggml_tensor *u_var = ggml_scale(ctx, sum_rows, dnm);

    // 求样本标准差(Sample Standard Deviation)
    ggml_tensor *standardize = ggml_sqrt(ctx, u_var);

    ggml_tensor *normalize = ggml_div(ctx, base_line, standardize);

    ggml_tensor *est = ggml_mul_mat(ctx, normalize, normalize);

    float scale = 1.0f / ((float) sample_set->ne[0] - 1.0f);
    ggml_tensor *conv = ggml_scale(ctx, est, scale);

    return conv;
}

int main(int argc, char **argv) {
    struct ggml_init_params params = ggml_init_params();
    params.mem_size = 1024 * 1024 * 1204;
    ggml_context *ctx = ggml_init(params);
    ggml_cgraph *graph = ggml_new_graph(ctx);

    const int64_t ne[4] = {7, 3};
    ggml_tensor *matrix = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);
    fill_rand_f32(matrix, 256.0);

    // ggml_set_f32(matrix, 3.14);
    ggml_build_forward_expand(graph, matrix);

    insight_matrix_tensor_f32(matrix, "matrix");

    ggml_tensor *tensor_cov = tensor_covariance(ctx, matrix);
    ggml_build_forward_expand(graph, tensor_cov);
    ggml_graph_compute_with_ctx(ctx, graph, 2);

    insight_matrix_tensor_f32(tensor_cov, "covariance");

    ggml_free(ctx);
    return 0;
}