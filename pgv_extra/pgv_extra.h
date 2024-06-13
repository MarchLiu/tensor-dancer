//
// Created by 刘鑫 on 2024/6/12.
//

#ifndef TENSOR_DANCER_PGV_EXTRA_H
#define TENSOR_DANCER_PGV_EXTRA_H
#include "postgres.h"

// code from pg vector
typedef struct Vector
{
    int32		vl_len_;		/* varlena header (do not touch directly!) */
    int16		dim;			/* number of dimensions */
    int16		unused;			/* reserved for future use, always zero */
    float		x[FLEXIBLE_ARRAY_MEMBER];
}			Vector;


#endif //TENSOR_DANCER_PGV_EXTRA_H
