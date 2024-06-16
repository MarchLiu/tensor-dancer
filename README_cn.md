# Tensor Dancer

Tensor Dancer（张量舞者）是一个高性能计算库，它以 BLAS/LAPACK 和 GGML 为核心，围绕 GGML 
的计算图构造一组高性能计算功能，重点关注为 PostgreSQL 等软件环境引入高阶代数结构的高性能计算
支持。

## 主要功能

目前 Tensor Dancer 还处于刚刚开始开发的阶段，第一个子项目 `pvg_extra` 展示了通过 bytea
类型以矩阵乘法，处理 PG Vector 向量的能力。

后续计划增加 Matrix 与 [PG Vector](https://github.com/pgvector/pgvector) 向量的类型转换、ggml tensor 与 vector 的转换，基于
vector 数据集直接进行 pca 分析等功能。

正在开发中的 `tensor_dancer` 项目，将会首先将 ggml 的 tensor 类型，以及 tensor dancer
的 matrix 类型引入 postgresql，然后定义各种算法的 pg 函数接口。

## 技术栈

Tensor Dancer 以 BLAS 、 LAPACK 和 GGML 为算法内核，在用于支持 [PG Vector](https://github.com/pgvector/pgvector) 的函数中，使用
了 [PG Vector](https://github.com/pgvector/pgvector)

## 向 Tensor Dancer 项目捐赠

<form action="https://www.paypal.com/cgi-bin/webscr" method="post" target="_top">
<input type="hidden" name="cmd" value="_donations">
<input type="hidden" name="business" value="march.liu@gmail.com">
<input type="hidden" name="item_name" value="Donate to this project">
<input type="hidden" name="currency_code" value="USD">
<input type="image" src="https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif" border="0" name="submit" alt="PayPal - The safer, easier way to pay online!">
<img alt="" border="0" src="https://www.paypal.com/en_US/i/scr/pixel.gif" width="1" height="1">
</form>