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

## About Me

My name is Liu Xin, and my English name is Mars Liu and previously used March Liu. I translated the Python
2.2/2.3/2.4/2.5/2.7 Tutorial under this pseudonym.

In recent years, I published a book titled "Construction and Implementation of Micro Lisp Interpreter", which is based
on my Jaskell Core library ([https://github.com/MarchLiu/jaskell-core](https://github.com/MarchLiu/jaskell-core)). The
book introduces some knowledge about interpreter development.

I am one of the earliest users in both the Python Chinese Community and PostgreSQL Chinese Community. At QCon, I
demonstrated a neural network algorithm implemented using SQL CTE
syntax: [SQL CTE](https://github.com/MarchLiu/qcon2019shanghai/tree/master/sql-cte).

## Donate

[![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/paypalme/marsliuzero)