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
了 [PG Vector](https://github.com/pgvector/pgvector) 。

项目主要使用 C 和 CPP，使用 meson 构建工具，有少量使用 python 编写的工具代码。使用 meson 是因为 PostgreSQL 使用 meson
构建，基于同样的理由，未来可能会有一部分使用 perl 的测试脚本。

## 关于我

我叫刘鑫，英文名 Mars Liu，以前我也用过 March Liu ，并以这个署名翻译了 Python 2.2/2.3/2.4/2.5/2.7 版的《Python
Tutorial》。

前几年，我出版了《微型 Lisp
解释器的构造与实现》，这本书基于我的组合子库 [Jaskell Core](https://github.com/MarchLiu/jaskell-core) ，介绍了
一些解释器开发的知识。

我是 Python 中文社区和 PostgreSQL 中文社区最早的用户，在 QCon 活动中，我演示过用 SQL CTE 语法实现的神经网络算法：
[SQL CTE](https://github.com/MarchLiu/qcon2019shanghai/tree/master/sql-cte)

## 捐助这个项目

您的赞助会使这个项目更健康的成长。

[![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/paypalme/marsliuzero)