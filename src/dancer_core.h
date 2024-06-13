//
// Created by 刘鑫 on 2024/6/10.
//

#ifndef TENSOR_DANCER_DANCER_CORE_H
#define TENSOR_DANCER_DANCER_CORE_H

#include <vector>
#include <utility>
#include <queue>
#include "ggml.h"

#ifdef USE_PG
#include "postgres.h"
#define dalloc(size) palloc(size);
#define dfree(data) pfree(data);
#else
#define dalloc(size) malloc(size);
#define dfree(data) free(data);
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct MatrixHeader {
    ggml_type type;
    size_t rows;
    size_t columns;
};

struct Matrix {
    Matrix();

    virtual ~Matrix();

    unsigned int magic;
    ggml_type type;
    size_t rows;
    size_t columns;
    void *data;
};

int load_matrix(const char *filename, Matrix &matrix);

int fill_matrix(std::istream &input, Matrix &matrix);

int mul_matrix_vector_f32(Matrix &matrix, float * vector, float * result);

#ifdef __cplusplus
};
#endif

template<typename T>
struct Indexed {
    int index;
    T element;
};

template<typename T>
bool compare(const T &a, const T &b) {
    return a > b;
}

template<typename T>
struct compare_indexed {
    bool operator()(const Indexed<T> &a, const Indexed<T> &b) const {
        return a.element > b.element;
    }
};


template<typename T>
std::vector<Indexed<T>> top_k_indexed(std::vector<T> data, size_t k) {
    if (k <= 0 || k > data.size()) {
        throw std::invalid_argument("k should be between 1 and the size of the data vector");
    }

    std::priority_queue<Indexed<T>, std::vector<Indexed<T>>, compare_indexed<T>> maxHeap;

    for (auto it = data.begin(); it != data.end(); ++it) {
        Indexed<T> pair;
        pair.index = std::distance(data.begin(), it);
        pair.element = *it;
        maxHeap.push(pair);
        // 如果堆的大小超过k，弹出堆顶元素
        if (maxHeap.size() > k) {
            maxHeap.pop();
        }
    }


    std::vector<Indexed<T>> result;
    while (!maxHeap.empty()) {
        result.push_back(maxHeap.top());
        maxHeap.pop();
    }

    std::reverse(result.begin(), result.end());
    return result;
}

#endif //TENSOR_DANCER_DANCER_CORE_H
