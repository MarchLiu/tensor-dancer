//
// Created by 刘鑫 on 2024/6/12.
//

#include "pgv_extra.h"
#include "fmgr.h"
#include "dancer.h"
#include <varatt.h>
#include <array.h>

#define VECTOR_SIZE(_dim)        (offsetof(Vector, x) + sizeof(float)*(_dim))
#define DatumGetVector(x)        ((Vector *) PG_DETOAST_DATUM(x))
#define PG_GETARG_VECTOR_P(x)    DatumGetVector(PG_GETARG_DATUM(x))
#define PG_RETURN_VECTOR_P(x)    PG_RETURN_POINTER(x)
#define BYTEA_SIZE(x)   (VARSIZE(x) - VARHDRSZ)
/*
 * code from pg vector
 * Allocate and initialize a new vector
 */
Vector *
InitVector(int dim) {
    Vector *result;
    int size;

    size = VECTOR_SIZE(dim);
    result = (Vector *) palloc0(size);
    SET_VARSIZE(result, size);
    result->dim = dim;

    return result;
}

PG_MODULE_MAGIC;

PGDLLEXPORT PG_FUNCTION_INFO_V1(pgv_mulmv);

Datum
pgv_mulmv(PG_FUNCTION_ARGS) {
    bytea *a = PG_GETARG_BYTEA_P(0);

    Vector *b = PG_GETARG_VECTOR_P(1);


    char *data = a->vl_dat;
    int len = BYTEA_SIZE(a);
    struct Matrix *matrix = InitMatrixF32();
    // todo assert status
    int status = write_matrix(matrix, data, len);

    char *message = palloc(256);
    ereport(INFO, (errcode(ERRCODE_DATA_EXCEPTION),
            errmsg("matrix load %d data get status %d, rows %zu, columns %zu",
                   len, status, matrix->rows, matrix->columns)));

    Vector *result = InitVector((int) matrix->rows);

    // todo assert status
    status = mul_matrix_vector_f32(matrix, b->x, result->x);

    // pfree(matrix->data);
    pfree(message);
    FreeMatrix(matrix);
    PG_RETURN_VECTOR_P(result);
}
