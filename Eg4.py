import numpy as np
from matplotlib import pyplot as plt

'''
计算方法A_2班(令丹老师)：上级作业第4题：
龙贝格数值积分
作者：陈长
学号：3120104099
日期：20201209
'''


# -----------------------------例题输入区-START--------------------------
# # 例题4.1
# def f(x):
#     return 1./(1.+x)
# a = np.float64(0)
# b = np.float64(1)
# T = (b-a) / 2 * (f(a) + f(b))  # first T
# 例题4.2
# def f(x):
#     return np.log(1+x)/(1+x**2)
# a = np.float64(0)
# b = np.float64(1)
# T = (b-a) / 2 * (f(a) + f(b))  # first T
# 例题4.3
# def f(x):
#     return np.log(1+x)/x
# a = np.float64(0)
# b = np.float64(1)
# T = (b-a) / 2 * (1 + f(b))  # first T
# # 例题4.4
def f(x):
    return np.sin(x) / x
a = np.float64(0)
b = np.float64(np.pi / 2)
T = (b - a) / 2 * (1 + f(b))  # first T
# -----------------------------例题输入区-END----------------------------

h = b - a
TArray = np.array(T)
# TArray = np.append(TArray,b)

for k in range(4):  # generate T start series
    TNew = T / 2 + h / 2 ** (k + 1) * np.sum(f(a + (2 * (np.arange(1, 2 ** k + 1)) - 1) * h / 2 ** (k + 1)))
    TArray = np.append(TArray, TNew)
    T = TNew

SArray = np.diff(TArray)
i = np.arange(k + 1)
SArray[i] = TArray[i + 1] + 1 / (4 - 1) * (TArray[i + 1] - TArray[i])
CArray = np.diff(SArray)
i = np.arange(k)
CArray[i] = SArray[i + 1] + 1 / (4 ** 2 - 1) * (SArray[i + 1] - SArray[i])
RArray = np.diff(CArray)
i = np.arange(k - 1)
RArray[i] = CArray[i + 1] + 1 / (4 ** 3 - 1) * (CArray[i + 1] - CArray[i])
while np.abs(RArray[-1] - RArray[-2]) > 10 ** (-12):
    k += 1  # k = 4
    TNew = T / 2 + h / 2 ** (k + 1) * np.sum(f(a + (2 * (np.arange(1, 2 ** k + 1)) - 1) * h / 2 ** (k + 1)))
    TArray = np.append(TArray, TNew)
    T = TNew
    SArray = np.append(SArray, TArray[-1] + 1 / (4 - 1) * (TArray[-1] - TArray[-2]))
    CArray = np.append(CArray, SArray[-1] + 1 / (4 ** 2 - 1) * (SArray[-1] - SArray[-2]))
    RArray = np.append(RArray, CArray[-1] + 1 / (4 ** 3 - 1) * (CArray[-1] - CArray[-2]))

print('数值积分结果：', RArray[-1])
print('TArray:', TArray)
print('SArray:', SArray)
print('CArray:', CArray)
print('RArray:', RArray)

# # 绘图模块
fig, axs = plt.subplots(1)
axs.plot(np.arange(1, RArray.size), np.abs(RArray[:-1] - RArray[-1]), 'r')
axs.bar(np.arange(1, RArray.size), np.abs(RArray[:-1] - RArray[-1]))  # 绘制原函数曲线图，采用黄色，线宽较宽，以便与后续插值曲线对比
axs.grid(True)
axs.set_yscale('log')
axs.set_xlabel('iteration times')
axs.set_ylabel('abs(R[:-1]-R[-1])')
axs.set_title('eg4.4: Convergence Charateristic')
plt.show()
