from numpy import array, zeros, ones, linalg, mat, shape, arange, eye, exp, mean, var
import matplotlib.pyplot as plt


def load_data_set(file_name):
    data_mat = []
    label_mat = []
    with open(file_name) as fr:
        for line in fr.readlines():
            line = line.strip().split('\t')
            data_mat.append([float(x) for x in line[0: -1]])
            label_mat.append(float(line[-1]))
    return data_mat, label_mat


def ridge_regression(x_arr, y_arr, lam=0.2):
    """
        岭回归(ridge regression)
        如果特征比样本点还多(c > r)，也就是说输入数据的矩阵x不是满秩矩阵，非满秩矩阵在求逆时会出现问题。岭回归是一种缩减方法。
        岭回归就是在矩阵XTX上加一个λI从而使的矩阵非奇异，进而能对(XTX + λI)求逆。
        注：I矩阵是一个c*c的单位矩阵，对角线元素全为1
        这里通过引入λ来限制所有w之和，通过引入该惩罚项，能够减少不重要的参数，这个技术在统计学中也叫缩减(shringkage)
    :param x_arr: 数据集
    :param y_arr: 结果集
    :param lam: lambda
    :return:
    """
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    r, c = shape(x_mat)

    xTx = x_mat.T * x_mat
    denom = xTx + eye(c) * lam

    if linalg.det(denom) == 0.0:  # 计算行列式，如果行列式非零，即该矩阵满秩。
        print('+++++this matrix is singular, cannot do inverse!')
        return
    weights = denom.I * x_mat.T * y_mat.T
    return weights


def ridge_test(x_arr, y_arr):
    """
        数据标准化的过程。
        两种常用的归一化方法：
        第一种：min-max标准化（Min-Max Normalization）
            也称为离差标准化，是对原始数据的线性变换，使结果值映射到[0 - 1]之间。
            x = (x - min) / (max - min)
        第二种：Z-score标准化方法
            这种方法给予原始数据的均值（mean）和标准差（standard deviation）进行数据的标准化。
        经过处理的数据符合标准正态分布，即均值为0，标准差为1，转化函数为：
            x = (x - μ) / δ
            μ: 所有样本数据的均值
            δ: 所有样本数据的标准差
    :param x_arr: 数据集
    :param y_arr: 结果集
    :return:
    """
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    y_mean = mean(y_mat, 0)
    y_mat = y_mat - y_mean
    x_means = mean(x_mat, 0)
    x_variance = var(x_mat, 0)  # 方差
    x_mat = (x_mat - x_means) / x_variance
    num_test_pts = 30
    weights = zeros((num_test_pts, shape(x_mat)[1]))
    for i in range(num_test_pts):
        ws = ridge_regression(x_mat, y_mat, exp(i-10))
        weights[i, :] = ws.T
    return weights


def show_data(ridge_weights):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridge_weights)

    plt.show()


if __name__ == '__main__':
    d_m, l_m = load_data_set('./data/abalone.txt')
    weights = ridge_test(d_m, l_m)
    show_data(weights)




