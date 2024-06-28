//
// Created by 刘鑫 on 2024/6/14.
//

#include "tensor_dancer.h"
#include "src/dancer.h"
#include "postgres.h"
#include "fmgr.h"
#include "copilota.h"
#include <builtins.h>
#include <executor/spi.h>

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

PGDLLEXPORT PG_FUNCTION_INFO_V1(copilota);

Datum
copilota(PG_FUNCTION_ARGS) {
    text *ask = PG_GETARG_TEXT_P(0);

    char *question = text_to_cstring(ask);

    SPI_connect();
    const char *answer = agent(question);
    SPI_finish();

    text* result = cstring_to_text(answer);
    PG_RETURN_TEXT_P(result);
}


#ifdef __cplusplus
};
#endif
