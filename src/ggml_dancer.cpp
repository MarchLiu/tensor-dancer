//
// Created by 刘鑫 on 2024/6/10.
//
#include <cassert>
#include "ggml.h"
#include "ggml_dancer.h"
#include "lapacke.h"
#include "dancer.h"

#ifndef NDEBUG
#include "insight.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

// 将矩阵形式的张量分解为奇异值矩阵和右奇矩阵
int lapack_svd_rf32(ggml_tensor *tensor_A, ggml_tensor *tensor_S, ggml_tensor *tensor_VT) {
    // float *A, int m, int n
    auto *A = (float *) tensor_A->data;
    int m = static_cast<int>(tensor_A->ne[0]);
    int n = static_cast<int>(tensor_A->ne[1]);

#ifdef DEBUG
    for (int64_t i = 0; i < tensor_A->ne[0]; i++) {
        for(int64_t j = 0; j < tensor_A->ne[1]; j++) {
            float value = ggml_get_f32_nd(tensor_A, i, j, 0, 0);
            if(isnan(value)) {
                printf("invalid data %lld, %lld\n", i, j);
            }
        }
    }
#endif

    auto S = (float *) tensor_S->data;
    float *U = nullptr;
    auto *VT = (float *) tensor_VT->data;
    // 查询所需工作空间的大小
    float work_query;
    int lwork = -1;
    int info;
    LAPACKE_sgesvd_work(LAPACK_ROW_MAJOR, 'N', 'A', m, n, A, m, S,
                        U, m, VT, n, &work_query, lwork);

    // 根据查询结果分配工作空间
    lwork = (int) work_query;
    auto *work = (float *) dalloc(sizeof(double) * lwork);
    if (work == nullptr) {
        fprintf(stderr, "Memory allocation failed for work array.\n");
        return -1;
    }

    // 执行奇异值分解
    info = LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'N', 'A', m, n, A, m, S,
                          U, m, VT, n, work);

    // 检查LAPACK函数是否成功执行
    if (info != 0) {
        fprintf(stderr, "LAPACKE_dgesvd returned with error code %d.\n", info);
        dfree(work);
        return info;
    }

    // 清理工作空间
    dfree(work);
    return 0;
}

// 中心化
ggml_tensor *dancer_centralization(ggml_context *ctx, ggml_tensor *tensor_X) {
    // 求均值
    ggml_tensor *mean = ggml_mean(ctx, tensor_X);

    struct ggml_tensor *center = ggml_repeat(ctx, mean, tensor_X);

    return ggml_sub(ctx, tensor_X, center);
}

// 方差
ggml_tensor *dancer_variance(ggml_context *ctx, ggml_tensor *tensor_X) {

    // 中心化
    ggml_tensor *base_line = dancer_centralization(ctx, tensor_X);

    // 求方差
    ggml_tensor *sqr = ggml_sqr(ctx, base_line);
    ggml_tensor *sum_rows = ggml_sum_rows(ctx, sqr);
    auto dnm = 1.0f / ((float) tensor_X->ne[0]);
    return ggml_scale(ctx, sum_rows, dnm);
}

// 总体标准差
ggml_tensor *dancer_psdv(ggml_context *ctx, ggml_tensor *tensor_X) {
    ggml_tensor *var = dancer_variance(ctx, tensor_X);
    return ggml_sqr(ctx, var);
}

// 无偏标准差
ggml_tensor *dancer_usdv(ggml_context *ctx, ggml_tensor *tensor_X) {
    ggml_tensor *u_var = dancer_unbiased_variance(ctx, tensor_X);
    return ggml_sqr(ctx, u_var);
}

// 无偏方差
ggml_tensor *dancer_unbiased_variance(ggml_context *ctx, ggml_tensor *tensor_X) {
    // 求特征平均值
    ggml_tensor *mean = ggml_mean(ctx, tensor_X);

    struct ggml_tensor *center = ggml_repeat(ctx, mean, tensor_X);

    // 中心化
    ggml_tensor *base_line = ggml_sub(ctx, tensor_X, center);

    // 求无偏方差
    ggml_tensor *sqr = ggml_sqr(ctx, base_line);
    ggml_tensor *sum_rows = ggml_sum_rows(ctx, sqr);
    auto dnm = 1.0f / ((float) tensor_X->ne[0] - 1.0f);
    return ggml_scale(ctx, sum_rows, dnm);
}

// 对样本集做规范化处理
// 为了不重复构造执行图的中间部分，这里没有调用中心化操作无偏方差的实现，而是整合在一起
// ggml 内置的 norm 算法使用了总体标准差，而dancer norm 使用无偏标准差
ggml_tensor *dancer_normalized(ggml_context *ctx, ggml_tensor *tensor_X) {

    // 求特征平均值
    ggml_tensor *mean = ggml_mean(ctx, tensor_X);
    ggml_set_name(mean, "Mean");

    struct ggml_tensor *center = ggml_repeat(ctx, mean, tensor_X);
    ggml_set_name(center, "Centiliter");

    // 中心化
    ggml_tensor *base_line = ggml_sub(ctx, tensor_X, center);
    ggml_set_name(base_line, "Centillion");

    // 求无偏方差
    ggml_tensor *sqr = ggml_sqr(ctx, base_line);
    ggml_set_name(sqr, "SQR");
    ggml_tensor *sum_rows = ggml_sum_rows(ctx, sqr);
    ggml_set_name(sum_rows, "SUM");
    auto dnm = 1.0f / ((float) tensor_X->ne[0] - 1);
    ggml_tensor *u_var = ggml_scale_inplace(ctx, sum_rows, dnm);
    ggml_set_name(u_var, "Unbiased Variance");

    // 求样本标准差(Sample Standard Deviation)
    ggml_tensor *standardize = ggml_sqrt_inplace(ctx, u_var);
    ggml_set_name(standardize, "Standardized");
    // 防止零除错误，对标准差做补偿
//    ggml_tensor* epss = ggml_new_f32(ctx, DANCER_EPS);
//    ggml_set_name(epss, "EPS Scalar Value");
//    ggml_tensor* epst = ggml_repeat(ctx, epss, standardize);
//    ggml_set_name(epst, "EPS Scalar Value Scaled");
//    ggml_tensor* std_dev = ggml_add_inplace(ctx, standardize, epst);
//    ggml_set_name(std_dev, "Sample Standard Deviation");
    return ggml_div_inplace(ctx, base_line, standardize);
}

ggml_tensor *dancer_covariance(ggml_context *ctx, struct ggml_tensor *tensor_X) {
    ggml_tensor *norm = dancer_normalized(ctx, tensor_X);
    ggml_set_name(norm, "Normalize");

    ggml_tensor *est = ggml_mul_mat(ctx, norm, norm);
    ggml_set_name(est, "EST");

    float scale = 1.0f / ((float) tensor_X->ne[0] - 1.0f);
    return ggml_scale(ctx, est, scale);
}

// 求解样本集 X 的 k 维 pca 转换矩阵
// 这个算法是非惰性的，在内部构造一个一次性的执行图
// 因为解 sdv 要在 lapack 中进行，目前这个实现并不算很高效，如果要根本性的优化这个算法，需要在 ggml 环境中做 sdv 分解
// 因为目前是 lapack 实现，所以仅支持 f32 类型的样本集，生成的 pca 矩阵也是 f32 类型
ggml_tensor *dancer_pca_force(ggml_context *ctx, ggml_tensor *tensor_X, size_t k, int n_threads) {
    ggml_cgraph *graph = ggml_new_graph(ctx);

    ggml_tensor *cov = dancer_covariance(ctx, tensor_X);
    ggml_set_name(cov, "Covariance");
    ggml_build_forward_expand(graph, cov);
    ggml_graph_compute_with_ctx(ctx, graph, n_threads);

    int64_t s_ne[4] = {1, cov->ne[1], 1, 1};
    ggml_tensor *tensor_S = ggml_new_tensor(ctx, tensor_X->type, 2, s_ne);

    int64_t vt_ne[4] = {cov->ne[1], cov->ne[1], 1, 1};
    ggml_tensor *tensor_VT = ggml_new_tensor(ctx, tensor_X->type, 2, vt_ne);

    int status = lapack_svd_rf32(cov, tensor_S, tensor_VT);
    assert(status == 0);

#ifdef DEBUG
    ggml_tensor *norm = ggml_get_tensor(ctx, "Normalize");
    insight_matrix_tensor_f32(norm, "Normalize");
    insight_matrix_tensor_f32(cov, "COV");
    insight_matrix_tensor_f32(tensor_S, "S");
    insight_matrix_tensor_f32(tensor_VT, "VT");
#endif

    int64_t result_ne[4] = {static_cast<int64_t>(k), cov->ne[1], 1, 1};
    ggml_tensor *tensor_result = ggml_new_tensor(ctx, cov->type, 2, result_ne);

    for (int i = 0; i < tensor_result->ne[0]; i++) {
        for (int j = 0; j < tensor_result->ne[1]; j++) {
            float value = ggml_get_f32_nd(tensor_VT, i, j, 0, 0);
            ggml_set_f32_nd(tensor_result, i, j, 0, 0, value);
        }
    }

    ggml_graph_clear(graph);
    return tensor_result;
}



#ifdef __cplusplus
};
#endif