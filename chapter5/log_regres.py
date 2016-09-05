from numpy import exp, mat, shape, ones, array, arange, random
import matplotlib.pyplot as plt


def load_data_set():
    data_matrix = []
    label_matrix = []
    with open('data/testSet.txt', 'r') as f:
        for line in f.readlines():
            line_array = line.strip().split()
            data_matrix.append([1.0, float(line_array[0]), float(line_array[1])])
            label_matrix.append(int(line_array[2]))
    return data_matrix, label_matrix


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


def gradient_descent(data_matrix_in, class_labels):
    data_matrix = mat(data_matrix_in)  # 转换为NumPy矩阵数据类型
    label_matrix = mat(class_labels).transpose()  # 矩阵转置
    m, n = shape(data_matrix)  # m为矩阵的行数，n为矩阵的列数
    alpha = 0.001
    max_cycles = 500
    weights = ones((n, 1))  # data_matrix为m行n列，weights为n行1列
    for k in range(max_cycles):
        h = sigmoid(data_matrix * weights)  # 矩阵相乘：第一个矩阵每一列*第二个矩阵每一行。h为列向量=样本的个数
        error = (label_matrix - h)
        weights = weights + alpha * data_matrix.transpose() * error
    return weights


def stochastic_gradient_descent(data_matrix, class_labels):
    """
        随机梯度上升算法。
        step1:原材料：数据集的行r列c，alpha步长，初始化参数w
        step2:
    :param data_matrix:ndarray类型的数据,每列分别代表每个不同的特征，每一行是一个训练样本
    :param class_labels:
    :return:
    """
    r, c = shape(data_matrix)
    alpha = 0.01  # 向目标移动的步长
    weights = ones([c, ])
    for i in range(r):
        h = sigmoid(sum(data_matrix[i] * weights))  # ndarray类型相乘，每个位置上的数字相乘就ok
        error = class_labels[i] - h
        weights += alpha * error * data_matrix[i]
    return weights


def stochastic_gradient_descent1(data_matrix, class_labels, num_iter=150):
    """
        改进-随机梯度上升算法。
    :param data_matrix:
    :param class_labels:
    :return:
    """
    r, c = shape(data_matrix)
    weights = ones(c)
    for j in range(num_iter):
        data_index = list(range(r))
        for i in range(r):
            alpha = 4 / (1.0 + j + i) + 0.0001
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_matrix[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights += alpha * error * data_matrix[rand_index]
            del (data_index[rand_index])
        return weights


def plot_best_fit(weights):
    data_matrix, label_matrix = load_data_set()
    data_array = array(data_matrix)
    r, c = shape(data_array)
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []
    for i in range(r):
        if int(label_matrix[i]) == 1:
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
    y = (-weights[0] - weights[1] * x) / weights[2]  # 最佳拟合直线
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    data_array, labels = load_data_set()
    we = stochastic_gradient_descent(array(data_array), labels)
    print(type(we))
    plot_best_fit(we)
