//
// Created by 刘鑫 on 2024/6/19.
//
#include <string>
#include <iostream>
#include <sstream>
#include <iostream>
#include "libpq-fe.h"
#include "dancer.h"

#define exit_nicely() \
    if(file != nullptr){fclose(file);} \
    if(res != nullptr){PQclear(res);} \
    PQfinish(conn);   \
    exit(1);

void write_vector(const char *vector, FILE *file) {
    std::string literal = vector;
    size_t length = literal.size();
    auto items = literal.substr(1, length - 2);
    std::istringstream stream(items);
    std::string s;
    while (getline(stream, s, ',')) {
        float element = std::stof(s);
        fwrite(&element, sizeof(float), 1, file);
    }
    fflush(file);
}

int main(int argc, char **argv) {
    char *matrix_filename = argv[1];
    PGresult *res = nullptr;
    FILE *file = nullptr;
    int limit = 10000;
    if (argc > 2) {
        limit = std::stoi(argv[2]);
    }

    std::string uri = "postgresql://localhost/pgv";

    PGconn *conn = PQconnectdb(uri.c_str());
    if (PQstatus(conn) != CONNECTION_OK) {
        fprintf(stderr, "%s", PQerrorMessage(conn));
        exit_nicely();
    }
    std::cout << "connected!" << std::endl;

    union {
        int intVal;
        char binaryVal[sizeof(int)];
    } paramData;
    paramData.intVal = limit;

    int lval = htonl(limit);

    Oid paramTypes[1] = {23};
    const char *paramValues[1] = {(const char *) (&lval)};
    int paramLengths[1] = {sizeof(int)};
    int paramFormats[1] = {1};

    res = PQexecParams(conn,
                       "select id, embedding from items limit $1",
                       1,
                       paramTypes,
                       paramValues,
                       paramLengths,
                       paramFormats,
                       0);
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        fprintf(stderr, "SET failed: %s", PQerrorMessage(conn));
        exit_nicely();
    }

    int nFields = PQnfields(res);
    for (int i = 0; i < nFields; i++) {
        printf("%-15s %d", PQfname(res, i), PQftype(res, i));
    }
    printf("\n\n");
    static const int columns = 4096;
    const int rows = PQntuples(res);

    struct MatrixHeader header{};
    header.magic = MATRIX_MAGIC;
    header.type = GGML_TYPE_F32;
    header.rows = rows;
    header.columns = columns;

    file = fopen(matrix_filename, "w+");
    write_matrix_header(&header, file);

    for (int i = 0; i < rows; i++) {
        char * id = PQgetvalue(res, i, 0);

        printf("vector id %s\n", id);

        write_vector(PQgetvalue(res, i, 1), file);
    }


    fflush(file);
    fclose(file);
    PQclear(res);
    PQfinish(conn);
    printf("completed\n");
    return 0;
}