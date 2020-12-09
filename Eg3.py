import numpy as np
from matplotlib import pyplot as plt

'''
计算方法A_2班(令丹老师)：上级作业第3题：
三次样条插值/牛顿和拉格朗日插值。
作者：陈长
学号：3120104099
日期：20201209
'''
# -----------------------------例题输入区-START--------------------------
x = np.linspace(-2.5, 2.5, 1000, dtype=np.float64)  # 生成自变量散点
y = 1 / (1 + 25 * x ** 2)  # 被插函数曲线

xi = np.linspace(-1, 1, 11, dtype=np.float64)  # 11个点，10分段函数
yi = 1 / (1 + 25 * xi ** 2)

y0p = np.float64(0.07396449704142012)  # x = -1处的一阶导数
ynp = -y0p  # x = 1处的一阶导数
# -----------------------------例题输入区-END----------------------------


def band_diagonal_solve(a1, b1):
    '''
    a1 is the A matrix, b1 is the b vector in Ax = b, type with np.float64
    return: x
    author: Doorvanbei, email: 3468274604@qq.com
    '''
    a = a1.copy()
    b = b1.copy()
    n = (a.shape)[1]
    for k in range(n):  # forward
        c = a[k, k]  # normalization coefficient
        b[k] /= c
        a[k] /= c  # normalization of 1st line
        for i in range(k + 1, k + 2 if k != n - 1 else n):  # only do once, do not exe when k == n-1
            c = -a[i, k]
            b[i] += c * b[k]
            a[i] += c * a[k]

    for i in range(n - 2, 0 - 1, -1):  # backward
        s = np.float64(0.)
        for j in range(i + 1, i + 2):  # only do once
            s += a[i, j] * b[j]
        b[i] -= s
    return b


def LagIntp(xi, yi):
    '''
    Lagrange插值多项式求取
    :param xi: 已知点横坐标向量, 要求为float类型输入!
    :param yi: 已知点纵坐标向量, 要求为float类型输入!
    :return: 插值多项式
    '''
    d = xi.shape[0]  # 取得被插函数的点数
    m = np.array([1.], dtype=np.float64)  # 待卷积多项式
    for i in range(d):
        m = np.polymul(m, np.array([1., -xi[i]]))
    s = np.array([0.], dtype=np.float64)  # 待求和多项式
    for i in range(d):
        num, tmp = np.polydiv(m, np.array([1, -xi[i]]))  # tmp丢弃，num返回分子多项式
        num *= yi[i] / np.polyval(num, xi[i])  # 系数包括：分母、函数值
        s = np.polyadd(s, num)
    return s


def NewtonIntp(xi, yi):
    '''
    Newton插值多项式求取
    :param xi: 已知点横坐标向量, 要求为float类型输入!
    :param yi: 已知点纵坐标向量, 要求为float类型输入!
    :return: 插值多项式
    '''
    d = xi.shape[0]  # 取得被插函数的点数
    yt = np.copy(yi)  # 拷贝，避免yi被函数修改
    n = [np.array([1.], dtype=np.float64)]  # 常数项
    for i in range(1, d):
        nt = np.array([1.])
        for j in range(i):
            nt = np.polymul(nt, np.array([1., -xi[j]]))
        n.append(nt)  # 得到每一个多项式(不含系数)
        y = yt.copy()  # 临时暂存向量
        for j in range(i, d):  # 求取每个系数
            yt[j] = (y[j] - y[j - 1]) / (xi[j] - xi[j - i])
    s = np.array([0.])
    for i, e in enumerate(n):
        s = np.polyadd(s, e * yt[i])
    return s


print('xi:', xi)
print('yi:', yi)
d = xi.shape[0]  # 输入向量长度，设为 d
hi = np.diff(xi)  # hi = xi - xi-1, 长度为 d-1
dyi = np.diff(yi)  # dyi = yi - yi-1, 长度为 d-1
mui = hi[:-1] / (hi[:-1] + hi[1:])  # mui = hi/(hi + hi+1), 长度为 d-2
lbdi = 1 - mui  # lambdai = 1 - mui
di = 6 * np.diff(dyi / hi) / (hi[:-1] + hi[1:])  # di公式

A = 2 * np.eye(d)
v = np.arange(d - 2)
A[v + 1, v] = mui
A[-1, -2] = 1.
A[0, 1] = 1.
v += 1
A[v, v + 1] = lbdi  # 生成系数矩阵A

d0 = 6 / hi[0] * ((yi[1] - yi[0]) / hi[0] - y0p)
dn = 6 / hi[-1] * (ynp - (yi[-1] - yi[-2]) / hi[-1])
dv = np.empty(d)
dv[1:-1] = di
dv[0], dv[-1] = d0, dn  # 生成右端向量dv

M = band_diagonal_solve(A, dv)  # 二阶导数向量
# ----------------------------输出三次样条插值函数计算结果----------------------------
print('h vector:', hi)
print('M vector:', M)

# 使用牛顿和拉格朗日插值多项式进行对比计算
sn = NewtonIntp(xi, yi)
sl = LagIntp(xi, yi)
print('Sn(x):', sn)
print('SL(x):', sl)

yN = x.copy()  # 使用牛顿插值的多项式函数向量
snx = np.empty(d)
for i, e in enumerate(x):
    j = np.arange(d)
    snx[-(j + 1)] = e ** j
    yN[i] = np.dot(sl, snx)

# 绘图模块
fig, axs = plt.subplots(1, 2)
axs[0].plot(x, y, 'y', linewidth=3, label='Primary')  # 绘制原函数曲线图，采用黄色，线宽较宽，以便与后续插值曲线对比
axs[0].grid(True)
axs[0].scatter(xi, yi)  # 绘制插值节点散点图
axs[0].plot(x, yN, 'r', linewidth=1, label='Newton')  # 绘制牛顿插值多项式函数曲线图
axs[0].set_xlim(-1.1, 1.1)
axs[0].set_ylim(-1, 2)
axs[0].set_title('Newton & primary')
axs[0].legend()

axs[1].plot(x, y, 'y', linewidth=3, label='Primary')  # 绘制原函数曲线图，采用黄色，线宽较宽，以便与后续插值曲线对比
axs[1].grid(True)
axs[1].set_xlim(-1.1, 1.1)
axs[1].set_ylim(0, 1.05)
axs[1].scatter(xi, yi)  # 绘制插值节点散点图
for i in range(1, d):  # 绘制三次样条插值函数分段曲线图
    xseg = np.linspace(xi[i - 1], xi[i], 100, dtype=np.float64)  # 生成在一个区间上的自变量向量
    # 生成在对应区间上的函数值
    S1 = ((xi[i] - xseg) ** 3) / 6 / hi[i - 1] * M[i - 1] + ((xseg - xi[i - 1]) ** 3) / 6 / hi[i - 1] * M[i] + (
            yi[i - 1] - hi[i - 1] ** 2 / 6 * M[i - 1]) * (xi[i] - xseg) / hi[i - 1] + (
                 yi[i] - hi[i - 1] ** 2 / 6 * M[i]) * (xseg - xi[i - 1]) / hi[i - 1]
    axs[1].plot(xseg, S1, 'r', linewidth=1)  # 绘制S(x)曲线，采用红色，线宽较窄
axs[1].set_title('cubic spline & primary')
axs[1].legend()
plt.show()
