//
// Created by 刘鑫 on 2024/6/21.
//

#include "ggml_dancer.h"
#include <iostream>
#include "dancer.h"
#include "insight.h"

using namespace std;

int main(int argc, const char **argv) {
    string samples_filename = argv[1];

    FILE *samples_file = fopen(samples_filename.c_str(), "r");

    auto header = load_matrix_file_header(samples_file);

    cout << "matrix file: " << samples_filename << endl;
    cout << "rows: " << header.rows << endl;
    cout << "columns: " << header.columns << endl;

    size_t size = header.rows * header.columns * 8;
    ggml_init_params params{};
    params.no_alloc = false;
    params.mem_buffer = nullptr;
    params.mem_size = size;
    ggml_context *ctx = ggml_init(params);
    struct ggml_tensor *samples = load_matrix_as_tensor(ctx, samples_filename.c_str());
    ggml_set_name(samples, "Samples");

    struct ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, samples);
    struct ggml_tensor *sum = ggml_sum_rows(ctx, samples);
    ggml_build_forward_expand(graph, sum);
    ggml_graph_compute_with_ctx(ctx, graph, 1);
    insight_matrix_tensor_f32(sum, "SUM");
    ggml_graph_print(graph);
    size_t ctx_size = ggml_get_mem_size(ctx);
    cout << "context memory: " << ctx_size << endl;

    vector<double> data(header.columns);
    std::fill(data.begin(), data.end(), 0.0f);
    for (size_t i = 0; i < header.rows; i++) {
        for(size_t j = 0; j < header.columns; j++) {
            float value = ggml_get_f32_nd(samples, i, j, 0, 0);
            if(isnan(value)) {
                printf("invalid data %zu, %zu\n", i, j);
            }
        }
    }

    ggml_free(ctx);
    return 0;
}
