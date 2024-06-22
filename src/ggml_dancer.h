//
// Created by 刘鑫 on 2024/6/10.
//

#ifndef TENSOR_DANCER_GGML_DANCER_H
#define TENSOR_DANCER_GGML_DANCER_H

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef DANCER_EPS
#define DANCER_EPS 1e-10
#endif

int lapack_svd_rf32(ggml_tensor *tensor_A, ggml_tensor *tensor_S, ggml_tensor *tensor_VT);

// centralization
ggml_tensor *dancer_centralization(ggml_context *ctx, ggml_tensor *tensor_X);

// variance
ggml_tensor *dancer_variance(ggml_context *ctx, ggml_tensor *tensor_X);

// normalize
ggml_tensor *dancer_normalized(ggml_context *ctx, ggml_tensor *tensor_X);

// unbiased variance
ggml_tensor *dancer_unbiased_variance(ggml_context *ctx, ggml_tensor *tensor_X);

// population standard deviation
ggml_tensor *dancer_psdv(ggml_context *ctx, ggml_tensor *tensor_X);
// unbiased standard deviation
ggml_tensor *dancer_usdv(ggml_context *ctx, ggml_tensor *tensor_X);
// covariance
ggml_tensor* dancer_covariance(ggml_context *ctx, ggml_tensor * tensor_X);

// pca
ggml_tensor* dancer_pca_force(ggml_context *ctx, ggml_tensor * tensor_X, size_t k, int n_threads);

#ifdef __cplusplus
};
#endif

#endif //TENSOR_DANCER_GGML_DANCER_H
