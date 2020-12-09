import numpy as np
from matplotlib import pyplot as plt

'''
计算方法A_2班(令丹老师)：上级作业第2题：
共轭梯度法求解线性方程组(系数矩阵对称正定)，绘制收敛特性。
作者：陈长
学号：3120104099
日期：20201209
'''

# -----------------------------例题输入区-START--------------------------
N = 400 # 矩阵维度
a1 = -2 * np.eye(N)
n = np.arange(N - 1)
a1[n + 1, n] = 1
a1[n, n + 1] = 1 # 生成系数矩阵A
b1 = np.zeros(N)
b1[0] = b1[-1] = -1  # 生成右端向量b
# -----------------------------例题输入区-END----------------------------
def ConjGrad(a1, b1, x01):
    '''
    a1 is the A matrix, b1 is the b vector in Ax = b, x01 is the initial value, type with np.float64
    return: times, x
    '''
    a, b, x0 = a1.copy(), b1.copy(), x01.copy()
    b, x0 = b[np.newaxis, :].T, x0[np.newaxis, :].T
    err = np.array(0, dtype=np.float64)
    x = x0
    d = r = b - a @ x
    times = 0
    while 1:
        alpha = (r.T @ d) / (d.T @ a @ d)
        x += alpha * d
        times += 1
        r = b - a @ x
        if np.linalg.norm(r) < 1.0e-13:
            err = np.append(err, np.linalg.norm(r))
            print('ConjGrad Method Converge!')
            break
        err = np.append(err, np.linalg.norm(r))
        beta = -(r.T @ a @ d) / (d.T @ a @ d)
        d = r + beta * d
    err = np.delete(err, 0)
    return [times, x, err]


x0 = np.zeros(N)
times, x, err = ConjGrad(a1, b1, x0)  # get iteration times and solution x from myfun
print(times)
# print(x)
# print(err)

# 绘图模块
fig, axs = plt.subplots(1)
axs.bar(np.arange(err.size), err)
axs.plot(np.arange(err.size), err, 'r')
axs.grid(True)
axs.set_yscale('log')
axs.set_xlabel('iteration times')
axs.set_ylabel('norm(r)')
axs.set_title('eg2: Convergence Charateristic')
plt.show()
