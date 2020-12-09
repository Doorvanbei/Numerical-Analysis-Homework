import numpy as np

'''
计算方法A_2班(令丹老师)：上级作业第1题：
使用基于Givens/Householder的QR分解求解方程组/超定方程组
作者：陈长
学号：3120104099
日期：20201209
'''
# -----------------------------例题输入区-START--------------------------
# 例题1.1 : 方阵求解
A = np.array([[5, 4, 7, 5, 6, 7, 5], [4, 12, 8, 7, 8, 8, 6], [7, 8, 10, 9, 8, 7, 7], [5, 7, 9, 11, 9, 7, 5],
              [6, 8, 8, 9, 10, 8, 9], [7, 8, 7, 7, 8, 10, 10], [5, 6, 7, 5, 9, 10, 10]], dtype=np.float64)
b = np.array([39, 53, 56, 53, 58, 57, 52], dtype=np.float64)
# 例题1.2 : 求最小二乘解
# A = np.array([[1, 3, -3], [2, 1, -2], [1, 1, 1], [1, 2, -3]], dtype=np.float64)
# b = np.array([-1, 1, 3, 1], dtype=np.float64)
# -----------------------------例题输入区-END----------------------------


def QrDecmpGivens(A):
    Ax = A.copy()
    m, n = Ax.shape
    PMul = np.eye(m)
    for i in range(n):  # 当前处于A的第i列消去过程，同时对于P来说，上行指针也是i
        for j in range(i + 1, m):  # 下行指针
            nm = np.linalg.norm(np.array([Ax[i, i], Ax[j, i]]))
            c, s = Ax[i, i] / nm, Ax[j, i] / nm
            P = np.eye(m)
            P[i, i], P[i, j], P[j, i], P[j, j] = c, s, -s, c
            Ax, PMul = P @ Ax, P @ PMul
    return PMul.T, Ax  # return Q, R


def QrDecmpHouseHolder(A):
    Ax = A.copy()
    m, n = Ax.shape
    HMul = np.eye(m)
    for j in range(m - 1):
        x = Ax[j:, j]
        sigma = -np.sign(x[0]) * np.linalg.norm(x)
        e = np.zeros(m - j, dtype=np.float64)
        e[0] = 1
        w = (np.array([x - sigma * e])).T
        u = w / np.linalg.norm(w)
        H = np.eye(m)
        H[j:, j:] = np.eye(m - j) - 2 * (u @ (u.T))
        Ax, HMul = H @ Ax, H @ HMul
    return HMul.T, Ax  # return Q, R


def LinSolveBackward(R):
    Az = R.copy()
    m, n = Az.shape
    n -= 1  # m = 4, n = 2
    for i in range(n):  # 规格化
        Az[i, :] /= Az[i, i]
    a = Az[:n, :n]
    b = Az[:n, -1]
    for i in range(n - 2, 0 - 1, -1):  # backward
        s = np.float64(0)
        for j in range(i + 1, n):
            s += a[i, j] * b[j]
        b[i] -= s
    return b  # 解或最小二乘解


def QrLinSolve(A_mn, b_n):
    A = A_mn.copy()
    b = b_n.copy()
    b = np.array([b])
    b = b.T
    Az = np.hstack((A, b))
    Q, R = QrDecmpGivens(Az)
    x = LinSolveBackward(R)
    xs = x.copy()
    xs = np.array([xs])
    xs = xs.T
    r = A @ xs - b
    r = np.reshape(r, np.shape(r)[0])
    err = np.linalg.norm(r)
    return x, err


# 主程序部分
x, err = QrLinSolve(A, b)
print('x =', x)  # 打印解/最小二乘解
print('norm(r) =', err)  # 残向量的二范数，对于方阵的解为0

Q, R = QrDecmpGivens(A)  # 或 Q,R = QrDecmpHouseHolder(A)
print('Givens Q:', Q)
print('Givens R:', R)
