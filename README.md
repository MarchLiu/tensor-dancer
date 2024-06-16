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

## Donate to the Tensor Dancer Project

<div>
<form action="https://www.paypal.com/cgi-bin/webscr" method="post" target="_top">
<input type="hidden" name="cmd" value="_donations">
<input type="hidden" name="business" value="march.liu@gmail.com">
<input type="hidden" name="item_name" value="Donate to this project">
<input type="hidden" name="currency_code" value="USD">
<input type="image" src="https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif" border="0" name="submit" alt="PayPal - The safer, easier way to pay online!">
<img alt="" border="0" src="https://www.paypal.com/en_US/i/scr/pixel.gif" width="1" height="1">
</form>
</div>