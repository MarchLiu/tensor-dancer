import numpy as np

row_set = [
    [106.802,       226.46, 112.205,        ],
    [165.608,       195.776,        165.748,        ],
    [255.05,        176.901,        251.349,        ],
    [114.21,        145.445,        90.0014,        ],
    [57.3648,       71.2387,        159.616,        ],
    [21.1655,       86.0189,        188.598,        ],
    [190.332,       220.956,        178.881,        ],
]

# 假设我们有一组二维向量
data = np.array(row_set)

# 计算每个特征的均值
mean = np.mean(data, axis=0)

# 计算每个特征的标准差
std = np.std(data, axis=0, ddof=1)

# 规范化数据：减去均值后除以标准差
# 为了避免除以0，可以确保std不是0
normalized_data = (data - mean) / (std + 1e-10)  # 添加一个小的常数避免除以0

# 使用numpy.cov函数计算规范化后数据的协方差矩阵
# 由于已经规范化，我们期望协方差矩阵是对角线上的值都是1的矩阵
cov_matrix = np.cov(normalized_data, rowvar=False)


print("原始数据:")
print(data)
# print("均值向量集:")
# print(mean)
# print("标准差:")
# print(std)
# print("\n规范化后的数据:")
# print(normalized_data)
print("\n规范化数据的协方差矩阵:")
print(cov_matrix)
