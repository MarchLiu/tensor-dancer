# Tensor Dancer

Tensor Dancer is a high-performance computing library, which takes BLAS/LAPACK and GGML as its core, and constructs a
set of high-performance computing capabilities around the computational graph of GGML, focusing on introducing
high-performance computing support for advanced algebraic structures into software environments such as PostgreSQL.

## Main Features

At present, Tensor Dancer is still in the early stages of development. The first sub-project pvg_extra demonstrates the
ability to handle matrix multiplication and PG Vector vectors through the bytea type.

Subsequent plans include adding type conversions between Matrix and PG Vector vectors, conversions between ggml tensors
and vectors, and performing PCA analysis directly on vector datasets.

The tensor_dancer project under development will first introduce the ggml tensor type and the Tensor Dancer matrix type
into PostgreSQL, and then define various algorithmic pg function interfaces.

## Technology Stack

Tensor Dancer uses BLAS, LAPACK, and GGML as the algorithmic core. In the functions supporting PG Vector, it utilizes PG
Vector.

