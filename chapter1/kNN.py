from numpy import zeros, array, tile

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import operator


def classify0(inX, data_set, labels, k):
    """
        step1:将本条数据扩充至900行，计算到900个训练数据的距离
        step2:将距离大小转化成名次
        step3:取出最近邻的三个训练数据和其label
        step4:按照label出现次数选出最高的

    :param inX: 分类的输入向量,即测试数据0-100其中一个
    :param data_set: 训练样本集100-1000
    :param labels: 标签100-1000
    :param k: 选择最近邻居的数目
    :return:
    """
    data_set_size = data_set.shape[0]  # 用于训练的数据个数
    diff_mat = tile(inX, (data_set_size, 1)) - data_set  # 很关键，本条测试数据*900再和训练数据相减
    sq_diff_mat = diff_mat ** 2  # 以下三行求距离
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5  # 很重要：测试数据与训练数据的距离(900个距离)
    sorted_dist_indicies = distances.argsort()  # 使用K近邻算法进行排序，即把900个距离按照大小排序并设置成1-900整数
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dist_indicies[i]]  # 选择距离最小的k个点对应的标签,int类型
        class_count[vote_label] = class_count.get(vote_label, 0) + 1  # 如果vote_label存在dict中value+1，如果不存在默认0+1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)  # 如：[(1, 2), (3, 1)]
    return sorted_class_count[0][0]


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def file2matrix(filename='./data/datingTestSet2.txt'):
    with open(filename, 'r') as fr:
        array_lines = fr.readlines()  # 按行读取，放入list中
        len_of_lines = len(array_lines)  # 获取文件行数
        return_mat = zeros((len_of_lines, 3))  # 初始化一个len_of_lines行3列的矩阵
        class_label_vector = []
        index = 0
        for line in array_lines:
            line = line.strip()
            list_from_line = line.split('\t')  # 40920	8.326976	0.953952	largeDoses
            return_mat[index, :] = list_from_line[0:3]  # 文件中每一行数据放入矩阵(return_mat)的第index行中
            class_label_vector.append(int(list_from_line[-1]))  # 'largeDoses', 'smallDoses', 'didntLike'
            index += 1
        return return_mat, class_label_vector


def show_data(data_mat, label_list):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    data_1 = []
    data_2 = []
    data_3 = []
    len_data = len(label_list)
    for i in range(len_data):
        if label_list[i] == 1:
            data_1.append(data_mat[i, :])
        if label_list[i] == 2:
            data_2.append(data_mat[i, :])
        if label_list[i] == 3:
            data_3.append(data_mat[i, :])

    data_1 = array(data_1)
    data_2 = array(data_2)
    data_3 = array(data_3)
    ax.scatter(data_1[:, 0], data_1[:, 1], data_1[:, 2], color='g')
    ax.scatter(data_2[:, 0], data_2[:, 1], data_2[:, 2], color='r')
    ax.scatter(data_3[:, 0], data_3[:, 1], data_3[:, 2], color='b')

    ax.set_xlabel('fly miles')
    ax.set_ylabel('play game scale')
    ax.set_zlabel('consume ice cream')
    plt.show()


def auto_norm(data_set):
    """
        tile（A,reps）,数据A沿各个维度重复的次数
        A=[1,2]
        tile(A,(2,3))
        结果：[[1,2,1,2，1,2], [1,2,1,2,1,2]]

        reps的数字从后往前分别对应A的第N个维度的重复次数。
        如tile（A,2）表示A的第一个维度重复2遍；
        tile（A,(2,3)）表示A的第一个维度重复3遍，然后第二个维度重复2遍；
        tile（A,(2,2,3)）表示A的第一个维度重复2遍，第二个维度重复2遍，第三个维度重复2遍。
    :param data_set:
    :return:
    """
    min_val = data_set.min(0)  # 第一列最小值,即：[ 0.        0.        0.001156]
    max_val = data_set.max(0)
    ranges = max_val - min_val
    # norm_data_set = zeros(shape=data_set)  # 初始化
    m = data_set.shape[0]  # 矩阵第一维度的长度
    norm_data_set = data_set - tile(min_val, (m, 1))  # tile()函数将变量内容复制成输入矩阵的同型矩阵
    norm_data_set = norm_data_set / tile(ranges, (m, 1))
    return norm_data_set, ranges, min_val


def dating_class_test():
    ho_ration = 0.10
    dating_data_mat, dating_labels = file2matrix('data/datingTestSet2.txt')
    norm_mat, ranges, min_values = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ho_ration)
    error_count = 0.0
    for i in range(num_test_vecs):
        # numpy.array[row, start:end],表示取第row行且start<=索引<end
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :], dating_labels[num_test_vecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print("the total error rate is: %f" % (error_count / float(num_test_vecs)))


if __name__ == "__main__":
    d_m, c_m = file2matrix()
    show_data(d_m, c_m)
