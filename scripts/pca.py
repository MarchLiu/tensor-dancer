# 基于 sklearn 的 pca 降维
import json

from sklearn.decomposition import PCA
import numpy as np

import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
import argparse
import struct

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--offset", "-o",
                        dest='offset', default=0,
                        type=int)
arg_parser.add_argument("--limit", "-l",
                        dest='limit', default=10000,
                        type=int)
arg_parser.add_argument("--dump-rsv",
                        dest='rsv', default=None,
                        type=str,
                        required=False)
arg_parser.add_argument("--dump-sample",
                        dest='sample', default=None,
                        type=str,
                        required=False)
arg_parser.add_argument("--dump-expect",
                        dest='expect', default=None,
                        type=str,
                        required=False)


def save_matrix(matrix: list[list[float]], filename: str):
    """
    save matrix as dancer f32 matrix file
    :param matrix: list of list
    :param filename: save to
    :return:
    """
    MAGIC_CODE = 0
    ggml_type = 0  # GGML_TYPE_F32
    print(f"save matrix[{len(matrix), len(matrix[0])}] into {filename}")
    with open(filename, "wb") as f:
        m_bytes = struct.pack("<i", MAGIC_CODE)
        f.write(m_bytes)
        t_bytes = struct.pack("<i", ggml_type)
        f.write(t_bytes)
        rows = len(matrix)
        r_bytes = struct.pack('<Q', rows)
        f.write(r_bytes)
        cols = len(matrix[0])
        c_bytes = struct.pack('<Q', cols)
        f.write(c_bytes)
        for row in matrix:
            for column in row:
                bytes_of_float = struct.pack('<f', column)
                f.write(bytes_of_float)


opts = arg_parser.parse_args()

engine = sa.create_engine("postgresql+psycopg2://localhost/pgv")
session_maker = sessionmaker(bind=engine)

dataset = []

with session_maker() as session:
    for row in session.execute(sa.text("select id, embedding from items offset :offset limit :limit"),
                               {"offset": opts.offset, "limit": opts.limit}).fetchall():
        dataset.append(json.loads(row[1]))

X = np.array(dataset)
pca = PCA(n_components=256)
pca.fit(X)

components = np.array(pca.components_.tolist())
# create a sample
data = dataset[2]

if opts.sample is not None:
    save_matrix([data], opts.sample)
if opts.expect is not None:
    vector = np.array(dataset[2])
    reduced = np.dot(components, vector)
    for idx, item in enumerate(reduced):
        print(f"{idx}=>{item}")
    save_matrix([reduced.tolist()], opts.expect)

# print(reduced)
if opts.rsv is not None:
    matrix = pca.components_.tolist()
    save_matrix(matrix, opts.rsv)
# print(pca.transform(X))
