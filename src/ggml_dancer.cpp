//
// Created by 刘鑫 on 2024/6/10.
//
#include "ggml.h"
#include "ggml_dancer.h"
#include "lapacke.h"
#include "dancer.h"

#ifdef __cplusplus
extern "C" {
#endif


// 将矩阵形式的张量分解为奇异值矩阵和右奇矩阵
int lapack_svd_rf32(ggml_tensor *tensor_A, ggml_tensor* tensor_S, ggml_tensor* tensor_VT) {
    // float *A, int m, int n
    auto *A = (float *) tensor_A->data;
    int m = static_cast<int>(tensor_A->ne[0]);
    int n = static_cast<int>(tensor_A->ne[1]);

    float *S = (float *)tensor_S->data;
    float *U = nullptr;
    float *VT = (float *)tensor_VT->data;
    // 查询所需工作空间的大小
    float work_query;
    int lwork = -1;
    int info;
    LAPACKE_sgesvd_work(LAPACK_ROW_MAJOR, 'N', 'A', m, n, A, m, S,
                        U, m, VT, n, &work_query, lwork);

    // 根据查询结果分配工作空间
    lwork = (int) work_query;
    auto *work = (float *) dalloc(sizeof(float) * lwork);
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
ggml_tensor* dancer_centralization(ggml_context* ctx, ggml_tensor * tensor_X) {
    // 求均值
    ggml_tensor *mean = ggml_mean(ctx, tensor_X);

    struct ggml_tensor *center = ggml_repeat(ctx, mean, tensor_X);

    return ggml_sub(ctx, tensor_X, center);
}

// 方差
ggml_tensor* dancer_variance(ggml_context* ctx, ggml_tensor * tensor_X){

    // 中心化
    ggml_tensor *base_line = dancer_centralization(ctx, tensor_X);

    // 求方差
    ggml_tensor *sqr = ggml_sqr(ctx, base_line);
    ggml_tensor *sum_rows = ggml_sum_rows(ctx, sqr);
    auto dnm = 1.0f / ((float) tensor_X->ne[0]);
    return ggml_scale(ctx, sum_rows, dnm);
}

// 总体标准差
ggml_tensor * dancer_psdv(ggml_context *ctx, ggml_tensor * tensor_X) {
    ggml_tensor * var = dancer_variance(ctx, tensor_X);
    return ggml_sqr(ctx, var);
}

// 无偏标准差
ggml_tensor * dancer_usdv(ggml_context *ctx, ggml_tensor * tensor_X) {
    ggml_tensor * u_var = dancer_unbiased_variance(ctx, tensor_X);
    return ggml_sqr(ctx, u_var);
}

// 无偏方差
ggml_tensor* dancer_unbiased_variance(ggml_context* ctx, ggml_tensor * tensor_X) {
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
ggml_tensor *dancer_normalized(ggml_context *ctx, ggml_tensor *tensor_X) {
    // 求特征平均值
    ggml_tensor *mean = ggml_mean(ctx, tensor_X);

    struct ggml_tensor *center = ggml_repeat(ctx, mean, tensor_X);

    // 中心化
    ggml_tensor *base_line = ggml_sub(ctx, tensor_X, center);

    // 求无偏方差
    ggml_tensor *sqr = ggml_sqr(ctx, base_line);
    ggml_tensor *sum_rows = ggml_sum_rows(ctx, sqr);
    auto dnm = 1.0f / ((float) tensor_X->ne[0] - 1);
    ggml_tensor *u_var = ggml_scale(ctx, sum_rows, dnm);

    // 求样本标准差(Sample Standard Deviation)
    ggml_tensor *standardize = ggml_sqrt(ctx, u_var);

    return ggml_div(ctx, base_line, standardize);
}

ggml_tensor* dancer_covariance(ggml_context *ctx, ggml_tensor * tensor_X) {
    ggml_tensor * norm = dancer_normalized(ctx, tensor_X);
    ggml_tensor *est = ggml_mul_mat(ctx, norm, norm);

    float scale = 1.0f / ((float) tensor_X->ne[0] - 1.0f);
    return ggml_scale(ctx, est, scale);
}

#ifdef __cplusplus
};
#endif