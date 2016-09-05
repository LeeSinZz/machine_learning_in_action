from numpy import mat, exp, shape, ones, array, arange, linalg, zeros, random, sum
import matplotlib.pyplot as plt


def load_data_set():
    data_mat = []
    label_mat = []
    with open('./data/testSet.txt') as fr:
        for line in fr.readlines():
            line_arr = line.strip().split()
            data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])  # X0 = 1.0
            label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def sigmoid(score):
    return 1.0 / (1 + exp(-score))


def gradient_ascent(data_mat, label_mat):
    data_matrix = mat(data_mat)
    label_matrix = mat(label_mat).transpose()
    r, c = shape(data_matrix)
    alpha = 0.001
    max_cycles = 500
    weights = ones((c, 1))  # Xn = r * c, Wn = c * 1
    for k in range(max_cycles):
        h = sigmoid(data_matrix * weights)  # h：r * 1阶拟合值
        print(h)
        error = (label_matrix - h)
        weights += alpha * data_matrix.transpose() * error
    return weights


def stochastic_gradient_ascent(data_mat, label_mat):
    """
        随机梯度上升算法，效果并不好，只能分对1/3的数据。
    :param data_mat:
    :param label_mat:
    :return:
    """
    data_mat = array(data_mat)
    r, c = shape(data_mat)
    alpha = 0.01
    weights = ones(c)
    for i in range(r):
        h = sigmoid(sum(data_mat[i] * weights))
        error = label_mat[i] - h
        weights += alpha * error * data_mat[i]
    return weights


def stochastic_gradient_ascent_upgrade(data_mat, label_mat, iteration=150):
    r, c = shape(data_mat)
    data_mat = mat(data_mat)
    weights = ones((c, 1))
    for j in range(iteration):
        data_index = list(range(r))
        for i in range(r):
            alpha = 4 / (1.0 + i + j) + 0.01
            random_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_mat[random_index] * weights))
            error = label_mat[random_index] - h
            weights += (alpha * error * data_mat[random_index]).transpose()
            del(data_index[random_index])
    return weights


def gradient_descent(data_mat, label_mat):
    """
        机器学习基石中的梯度下降算法
        注：Yn 是{-1， 1}、
    :param data_mat:
    :param label_mat:
    :return:
    """
    data_matrix = mat(data_mat)
    label_matrix = mat(label_mat).transpose()
    r, c = shape(data_matrix)
    eta = 0.1126
    max_cycles = 500
    weights = ones((c, 1))
    for k in range(max_cycles):
        sco = zeros((1, c))
        for i in range(r):
                tem = sigmoid(-label_matrix[i] * data_matrix[i] * weights) * (-label_matrix[i] * data_matrix[i])
                sco += tem
        descent_direction = sco / r
        weights -= eta * descent_direction.transpose()
    return weights


def plot_best_fit(weight):
    # weight = wei.getA()  # matrix类型转成ndarray。和asarray()效果一样
    data_mat, label_mat = load_data_set()
    data_array = array(data_mat)
    r, c = shape(data_array)
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []
    for i in range(r):
        if int(label_mat[i]) == 1:
            x_cord1.append(data_array[i, 1])
            y_cord1.append(data_array[i, 2])
        else:
            x_cord2.append(data_array[i, 1])
            y_cord2.append(data_array[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
    ax.scatter(x_cord2, y_cord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weight[0] - weight[1] * x) / weight[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def sga_weights(data_mat, label_mat, iteration=150):
    """
        使用样本随机选择和alpha动态减少机制的随机梯度上升算法返回的weights系数，
        plot_sga_weight()函数画出系数的收敛示意图， 比采用固定alpha的方法收敛的速度更快。

    :param data_mat:
    :param label_mat:
    :param iteration:
    :return:
    """
    r, c = shape(data_mat)
    data_mat = mat(data_mat)
    weights = ones((c, 1))
    weights_arr = []
    for j in range(iteration):
        data_index = list(range(r))
        for i in range(r):
            weights_arr.append(array(weights))  # 必须矩阵转成数组形式
            alpha = 4 / (1.0 + i + j) + 0.01
            random_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_mat[random_index] * weights))
            error = label_mat[random_index] - h
            weights += (alpha * error * data_mat[random_index]).transpose()
            del(data_index[random_index])
    return weights_arr
def plot_sga_weight(weight_arr):
    fig = plt.figure(figsize=(16, 9))
    ax0 = fig.add_subplot(311)
    weight0 = array(weight_arr)[:, 0]  # 返回二维数组的第一列
    x = arange(1, shape(weight0)[0]+1)
    y = weight0
    ax0.plot(x, y)

    ax1 = fig.add_subplot(312)
    weight0 = array(weight_arr)[:, 1]
    x = arange(1, shape(weight0)[0] + 1)
    y = weight0
    ax1.plot(x, y)

    ax2 = fig.add_subplot(313)
    weight0 = array(weight_arr)[:, 2]
    x = arange(1, shape(weight0)[0] + 1)
    y = weight0
    ax2.plot(x, y)

    plt.xlabel('Iteration')
    plt.ylabel('weight0')
    #plt.ylim(-4, 14)
    plt.show()






if __name__ == '__main__':
    d_m, l_m = load_data_set()
    weights = sga_weights(d_m, l_m)
    plot_sga_weight(weights)

