//
// Created by 刘鑫 on 2024/6/16.
//
#include <iostream>
#include "dancer.h"

int main(int argc, char** argv) {
    unsigned int magic_code = 0;
    char * buffer = (char *)&magic_code;
    buffer[0] = 'T';
    buffer[1] = 'D';
    buffer[2] = 'M';
    buffer[3] = 'X';
    std::cout << "magic code: " << magic_code << std::endl;

    for(int i=0; i<4; i++){
        printf("magic code %d is 0x%X value %d\n", i, (char)buffer[i], (char)buffer[i]);
    }

    for(int i=0; i<4; i++){
        unsigned char c = magic_code % 256;
        printf("magic code mode %d is 0x%X\n", i, c);
        magic_code = magic_code >> 8;
    }
}