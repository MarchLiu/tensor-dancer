#include <stdio.h>
#include "ggml.h"
#include "insight.h"

int main(int argc, char **argv) {
    struct ggml_init_params params;

    params.mem_size = 1024 * 1024 * 1204;
    struct ggml_context *ctx = ggml_init(params);
    struct ggml_cgraph *graph = ggml_new_graph(ctx);

    const int64_t ne[4] = {7, 3};
    struct ggml_tensor *matrix = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);
    fill_rand_f32(matrix, 256.0F);

    // ggml_set_f32(matrix, 3.14);
    ggml_build_forward_expand(graph, matrix);
    insight_matrix_tensor_f32(matrix, "matrix");


    ggml_graph_compute_with_ctx(ctx, graph, 2);

    ggml_free(ctx);
    return 0;
}
