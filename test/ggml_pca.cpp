//
// Created by 刘鑫 on 2024/6/21.
//
#include <string>
#include "dancer.h"
#include "ggml_dancer.h"
#include <iostream>
#include <cassert>
#include "insight.h"

using namespace std;

int main(int argc, char **argv) {
    string samples_filename = argv[1];
    string output_filename = argv[2];
    size_t down_to = stoll(argv[3]);
    assert(argc == 4);

    FILE *samples_file = fopen(samples_filename.c_str(), "r");

    auto header = load_matrix_file_header(samples_file);

    cout << "matrix file: " << samples_filename << endl;
    cout << "rows: " << header.rows << endl;
    cout << "columns: " << header.columns << endl;

    size_t size = 96000 + header.rows * header.rows * 8;
    ggml_init_params params{};
    params.mem_size = size;
    ggml_context *ctx = ggml_init(params);
    struct ggml_tensor *samples = load_matrix_as_tensor(ctx, samples_filename.c_str());
    ggml_set_name(samples, "Samples");

    struct ggml_tensor *pca = dancer_pca_force(ctx, samples, down_to, 8);

    insight_matrix_tensor_f32(pca, "PCA");

    save_tensor_as_matrix(pca, output_filename.c_str());

    cout << "tensor file: " << output_filename << endl;
    cout << "rows: " << pca->ne[0] << endl;
    cout << "columns: " << pca->ne[1] << endl;

    fclose(samples_file);
    ggml_free(ctx);
    return 0;
}