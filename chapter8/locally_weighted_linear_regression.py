from numpy import array, zeros, ones, linalg, mat, shape, arange, eye, exp
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


def lwlr(test_point, x_arr, y_arr, k=1.0):
    """
        局部加权线性回归(Locally Weighted Linear Regression)
        标准线性回归：Ein = (Y - XT * W)^2 平方误差
        LWLR：w(i) * (yi - xiT * w)^2
            w(i)为r * r的对角线矩阵，为正态分布形状。全体数据对test_point点错误衡量有影响，越靠近test_point的点
        权重越大；k越大正太分布衰减越快，即越尖。

    :param test_point: 测试点
    :param x_arr: 数据集
    :param y_arr: 结果集
    :param k: 缩减指数
    :return:
    """
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    r, c = shape(x_mat)
    weights = mat(eye(r))
    for i in range(r):
        different_mat = test_point - x_mat[i, :]
        weights[i, i] = exp(different_mat * different_mat.T / (-2.0 * k ** 2))
    xTx = x_mat.T * weights * x_mat
    if linalg.det(xTx) == 0.0:  # 计算行列式，如果行列式非零，即该矩阵满秩。
        print('+++++this matrix is singular, cannot do inverse!')
        return
    weights = xTx.I * (x_mat.T * weights * y_mat)  # I:表示逆矩阵(inverse)
    return test_point * weights


def lwlr_test(test_arr, x_arr, y_arr, k=1.0):
    r, c = shape(test_arr)
    y_hat = zeros(r)
    for i in range(r):
        y_hat[i] = lwlr(test_arr[i], x_arr, y_arr, k)
    return y_hat


def show_data(x_arr, y_arr, y_hat):
    x_arr = array(x_arr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_arr[:, 1], y_arr, color='g')

    sort_index = mat(x_arr)[:, 1].argsort(0)  # 按照第二列大小排序得到索引序列
    x_sort = x_arr[sort_index][:, 0, :]
    ax.plot(x_sort[:, 1], y_hat[sort_index][:, 0])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


if __name__ == '__main__':
    d_m, l_m = load_data_set('./data/ex0.txt')
    y_h = lwlr_test(d_m, d_m, l_m, 0.01)
    show_data(d_m, l_m, y_h)








