import numpy as np

row_set = [
    [100.185, 213.984, 138.593, ],
    [176.465, 198.925, 238.06, ],
    [253.292, 118.28, 229.892, ],
    [130.756, 230.796, 99.0423, ],
    [80.2366, 123.545, 125.673, ],
    [44.9698, 239.991, 146.176, ],
    [220.305, 172.746, 10.3925, ],
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

# cov_matrix = cov_matrix / cov_matrix[0][0]

print("原始数据:")
print(data)
# print("均值向量集:")
# print(mean)
print("标准差:")
print(std)
print("\n规范化后的数据:")
print(normalized_data)
print("\n规范化数据的协方差矩阵:")
print(cov_matrix)
