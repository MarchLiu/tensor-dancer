# 基于 sklearn 的 pca 降维
import json

from sklearn.decomposition import PCA
import numpy as np

import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
import argparse
import struct
from commons import save_matrix

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
