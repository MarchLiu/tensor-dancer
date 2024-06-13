//
// Created by 刘鑫 on 2024/6/10.
//
#include <iostream>
#include <vector>
#include "src/dancer_core.h"
#include <random>
#include <cassert>

using namespace std;

template<typename T>
void print_vector(vector<T> data, const char *name, int columns = 1) {
    cout << name << ":" << endl;
    int rows = data.size() / columns;
    for (int i = 0; i < rows; i++) {
        cout << "\t";
        for (int j = 0; j < columns; j++) {
            cout << data[i * columns + j] << "\t";
        }
        cout << endl;
    }
}

template<typename T>
void print_vector(vector<Indexed<T>> data, const char *name, int columns = 1) {
    cout << name << ":" << endl;
    int rows = data.size() / columns;
    for (int i = 0; i < rows; i++) {
        cout << "\t";
        for (int j = 0; j < columns; j++) {
            auto item = data[i * columns + j];
            cout << "(" << item.index << ", " << item.element << ")\t";
        }
        cout << endl;
    }
}

int main(int argc, char **argv) {
    std::mt19937 generator(std::random_device{}());

    int range_start = 1;
    int range_end = 100;
    std::uniform_int_distribution<int> distribution(range_start, range_end - 1);

    int len = 9;
    vector<int> data = vector<int>();
    data.reserve(9);

    for (int i = 0; i < len; i++) {
        data.push_back(distribution(generator));
    }

    vector<Indexed<int>> top = top_k_indexed(data, 3);

    print_vector(data, "data set", 3);

    print_vector(top, "top k", 3);

    for (auto item: top) {
        auto x = std::max_element(data.begin(), data.end());
        assert(*x == item.element);
        cout << "top k check of " << std::distance(data.begin(), x) << ": " << *x << " passed" << endl;
        data.erase(x);
    }
    cout << "all check passed" << endl;

    return 0;
}