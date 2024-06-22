//
// Created by 刘鑫 on 2024/6/20.
//

#include <cstdio>
#include <iostream>
#include "dancer.h"

int main(int argc, char **argv) {
    char *filename = argv[1];
    auto header = new struct MatrixHeader();
    FILE *file = fopen(filename, "rb");

    fread(header, sizeof(struct MatrixHeader), 1, file);
    char *magic = (char *) (&header->magic);

    std::cout << "Magic Code: " << magic << std::endl;
    std::cout << "Type Code: " << header->type << std::endl;
    std::cout << "Rows: " << header->rows << std::endl;
    std::cout << "Columns: " << header->columns << std::endl;

    fclose(file);
    delete header;
    return 0;
}