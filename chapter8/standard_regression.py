from numpy import array, zeros, ones, linalg, mat, shape, arange
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


def stand_regression(x_arr, y_arr):
    """
        Ein = (Y - XT * W)^2 平方误差
        上式求导得：w_hat = (XT * X)^-1 * Y
    :param x_arr:
    :param y_arr:
    :return:
    """
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    xTx = x_mat.T * x_mat
    if linalg.det(xTx) == 0.0:  # 计算行列式，如果行列式非零，即该矩阵满秩。
        print('+++++this matrix is singular, cannot do inverse!')
    weights = xTx.I * (x_mat.T * y_mat)  # I:表示逆矩阵(inverse)
    return weights


def show_data(x_arr, y_arr, weights):
    x_arr = array(x_arr)
    weights = array(weights)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_arr[:, 1], y_arr, color='g')

    x = arange(0, 1.0, 0.05)
    y = weights[0] + weights[1] * x
    ax.plot(x, y)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


if __name__ == '__main__':
    d_m, l_m = load_data_set('./data/ex1.txt')
    weights = stand_regression(d_m, l_m)
    show_data(d_m, l_m, weights)








