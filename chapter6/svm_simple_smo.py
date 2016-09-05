"""
支持向量机(support vector machines)
其中的一种实现：序列最小优化(sequential minimal optimization, SMO)

我们希望找到离分隔超平面最近的点，确保他们离分隔面的距离尽可能远

"""
from numpy import random, mat, array, shape, zeros, ones, multiply, arange
import matplotlib.pyplot as plt


def load_data_set(file_name='./data/testSet.txt'):
    data_mat = []
    label_mat = []
    with open(file_name) as fr:
        for line in fr.readlines():
            line_arr = line.strip().split('\t')
            data_mat.append([float(line_arr[0]), float(line_arr[1])])
            label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


def select_j_rand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, l, h):
    if aj > h:
        aj = h
    if aj < l:
        aj = l
    return aj


def simple_smo(data_mat, label_mat, C, fault_tolerance, max_iter):
    """

    :param data_mat: 数据集
    :param label_mat: 类别标签
    :param C: 常数C
    :param fault_tolerance: 容错率
    :param max_iter: 退出前最大循环次数
    :return:
    """
    data_matrix = mat(data_mat)
    label_matrix = mat(label_mat).transpose()
    b = 0
    r, c = shape(data_matrix)
    alphas = mat(zeros((r, 1)))
    iteration = 0
    while iteration < max_iter:
        alpha_pairs_changed = 0
        for i in range(r):
            fitted_value_i = float(multiply(alphas, label_matrix).T * (data_matrix * data_matrix[i].T)) + b
            error_i = fitted_value_i - float(label_matrix[i])
            if ((label_matrix[i] * error_i < -fault_tolerance) and (alphas[i] < C)) or ((label_matrix[i] * error_i > fault_tolerance) and (alphas[i] > 0)):
                j = select_j_rand(i, r)
                fitted_value_j = float(multiply(alphas, label_matrix).T * (data_matrix * data_matrix[j].T)) + b
                error_j = fitted_value_j - float(label_matrix[j])
                alpha_i_old = alphas[i].copy()  # copy()之后还是二维数组
                alpha_j_old = alphas[j].copy()
                if label_matrix[i] != label_matrix[j]:
                    l = max(0, alphas[j] - alphas[i])
                    h = min(C, C + alphas[j] - alphas[i])
                else:
                    l = max(0, alphas[j] + alphas[i] - C)
                    h = min(C, alphas[j] + alphas[i])
                if l == h:
                    # print('++++++ l == h')
                    continue
                eta = 2.0 * data_matrix[i] * data_matrix[j].T - data_matrix[i] * data_matrix[i].T - data_matrix[j] * data_matrix[j].T
                if eta >= 0:
                    # print('+++++++ eta >= 0')
                    continue
                alphas[j] -= label_matrix[j] * (error_i - error_j) / eta
                alphas[j] = clip_alpha(alphas[j], l, h)
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    # print('++++++ j not moving enough')
                    continue
                alphas[i] += label_matrix[i] * label_matrix[i] * (alpha_j_old - alphas[j])
                b1 = b - error_i - label_matrix[i] * (alphas[i] - alpha_i_old) * data_matrix[i] * data_matrix[i].T - label_matrix[j] * (alphas[j] - alpha_j_old) * data_matrix[i] * data_matrix[j].T
                b2 = b - error_j - label_matrix[i] * (alphas[i] - alpha_i_old) * data_matrix[i] * data_matrix[j].T - label_matrix[j] * (alphas[j] - alpha_j_old) * data_matrix[j] * data_matrix[j].T
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
                # print('iteration: %d i : %d, pairs changed %d' % (iteration, i, alpha_pairs_changed))

        if alpha_pairs_changed == 0:
            iteration += 1
        else:
            iteration = 0
        # print('++++++ iteration number: %d.' % iteration)
    return b, alphas


def get_weights(alphas, data_mat, label_mat):
    data_matrix = mat(data_mat)
    label_matrix = mat(label_mat).T
    r, c = shape(data_matrix)
    w = zeros((c, 1))
    for i in range(r):
        w += multiply(alphas[i] * label_matrix[i], data_matrix[i].T)
    return w


def plot_best_fit():
    d_m, l_m = load_data_set()
    b, alphas = simple_smo(d_m, l_m, 0.6, 0.001, 40)
    alphas = array(alphas)
    weights = get_weights(alphas, d_m, l_m)
    d_m = array(d_m)
    r, c = shape(d_m)

    with open('./data/alphas.txt', 'w') as f:
        for i in range(r):
            f.write(str(alphas[i, 0]))
            f.write('\n')

    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []
    for i in range(r):
        if l_m[i] == float(1):
            x_cord1.append(d_m[i, 0])
            y_cord1.append(d_m[i, 1])
        else:
            x_cord2.append(d_m[i, 0])
            y_cord2.append(d_m[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
    ax.scatter(x_cord2, y_cord2, s=30, c='green')
    x = arange(-1.0, 12.0, 0.1)
    y = (weights[0] * (x - 5)) / weights[1]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.ylim(-8.0, 6.0)
    plt.show()




if __name__ == '__main__':
    # d_m, l_m = load_data_set()
    # bb, al = simple_smo(d_m, l_m, 0.6, 0.001, 40)
    # print('-------------')
    # print(bb)
    # print(al)
    plot_best_fit()
