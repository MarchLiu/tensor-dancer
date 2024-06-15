//
// Created by 刘鑫 on 2024/6/14.
//

#include "tensor_dancer.h"
#include "src/dancer.h"
#include "postgres.h"
#include "fmgr.h"

#ifdef __cplusplus
exten "C" {
#endif

static inline void CHECK_MATRIX_FILL(int status) {
    if (status != 0) {
        ereport(ERROR,
                (errcode(ERRCODE_DATA_EXCEPTION),
                        errmsg("fill failed, the bytea may be not a valid matrix buffer")));
    }
}

static inline void CHECK_RESULT(int status) {
    if (status != 0) {
        ereport(ERROR,
                (errcode(ERRCODE_DATA_EXCEPTION),
                        errmsg("compute failed")));
    }
}

PG_MODULE_MAGIC;

PGDLLEXPORT PG_FUNCTION_INFO_V1(matrix_rows);

Datum
matrix_rows(PG_FUNCTION_ARGS) {
    bytea *a = PG_GETARG_BYTEA_P(0);
    bytea *b = PG_GETARG_BYTEA_P(1);

    char *data = a->vl_dat;
    struct Matrix* matrix = palloc(sizeof(struct Matrix));
    // todo assert status
    int32 result;
    memcpy(&result, a->vl_len_, 4);
    void * buffer = a->vl_dat;
    int32 status = write_matrix(matrix, buffer, result);
    CHECK_MATRIX_FILL(status);

    // pfree(matrix->data);
    pfree(matrix);
    PG_RETURN_INT32(result);
}


#ifdef __cplusplus
};
#endif
