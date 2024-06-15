//
// Created by 刘鑫 on 2024/6/10.
//

#ifndef TENSOR_DANCER_DANCER_CORE_H
#define TENSOR_DANCER_DANCER_CORE_H

#include <vector>
#include <utility>
#include <queue>
#include "ggml.h"

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

int fill_matrix(struct Matrix &matrix, std::istream &input);

#endif //TENSOR_DANCER_DANCER_CORE_H
