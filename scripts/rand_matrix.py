"""usage:
python rand_matrix.py dump.matrix rows columns upside
"""
import sys
from sklearn.decomposition import PCA
import numpy as np
import random

from commons import save_matrix

filename = sys.argv[1]
rows = int(sys.argv[2])
columns = int(sys.argv[3])
upside = float(sys.argv[4])

samples = []

for i in range(rows):
    row = [random.random() * upside for j in range(columns)]
    samples.append(row)

save_matrix(samples, filename)