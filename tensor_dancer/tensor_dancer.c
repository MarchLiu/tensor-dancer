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

//Datum
//matrix_rows(PG_FUNCTION_ARGS) {
//    bytea *a = PG_GETARG_BYTEA_P(0);
//
//
//    PG_RETURN_INT32(result);
//}


#ifdef __cplusplus
};
#endif
