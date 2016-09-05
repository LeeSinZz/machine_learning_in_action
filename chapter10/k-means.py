from numpy import zeros, inf, array, min, max, sqrt, sum, power, shape, mat, nonzero, random, mean, column_stack
import matplotlib.pyplot as plt


def load_data_set(file_name):
    data_arr = []
    with open(file_name) as fr:
        for line in fr.readlines():
            line_arr = line.strip().split('\t')
            data_arr.append(list(map(float, line_arr))) # map()返回一个迭代器，需要用list转下
    return data_arr


def distance_euclidean(vector_a, vector_b):
    return sqrt(sum(power(vector_a - vector_b, 2)))


def random_centroids(data_set, k):
    data_arr = array(data_set)
    r, c = shape(data_arr)
    centroids = mat(zeros((k, c)))  # k个质心点
    for j in range(c):
        min_j = min(data_arr[:, j])
        range_j = float(max(data_arr[:, j] - min_j))
        centroids[:, j] = min_j + range_j * random.rand(k, 1)  # rand(k, 1)生成随机0-1之间k*1的矩阵
    return centroids


def k_means(data_set, k, distance_method=distance_euclidean, create_centroids_method=random_centroids):
    """
        k-means算法
        step1: 随机挑选k个质心点，将分簇信息放入cluster_assessment矩阵中；
        step2: 外层无限循环，直到所有的点不再有分簇变动；
        step3: 遍历每个数据点到k个质心点的距离，如果距离比cluster_assessment中记录的小，更新数据点所属的簇；
        step4: 根据新的分簇信息生成k个新的质心点。

        data_arr[nonzero(cluster_assessment[:, 0].A == center)[0]]
        cluster_assessment[:, 0].A 将cluster_assessment矩阵类型转成narray类型
        cluster_assessment[:, 0].A == center 返回一个cluster_assessment大小的矩阵，等于center是True。
        nonzero(cluster_assessment[:, 0].A == center)[0] 返回不等于零(True)的索引
        根据索引将data_arr中某个center簇元素过滤出来
    :param data_set: 数据集
    :param k: 分簇个数
    :param distance_method: 计算两点距离方法
    :param create_centroids_method: 生成质心点的方法
    :return: k个质心点、分簇信息
    """
    data_arr = array(data_set)
    r, c = shape(data_set)
    cluster_assessment = mat(zeros((r, 2)))  # 簇分配结果矩阵，[簇索引值, 存储误差]
    centroids = create_centroids_method(data_arr, k)  # k个质心点
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(r):
            min_distance = inf  # 无限大
            min_index = -1
            for j in range(k):
                distance_j_i = distance_method(centroids[j, :], data_arr[i, :])
                if distance_j_i < min_distance:
                    min_distance = distance_j_i
                    min_index = j
            if cluster_assessment[i, 0] != min_index:
                cluster_changed = True
            cluster_assessment[i, :] = min_index, min_distance ** 2
        for center in range(k):
            pts_in_cluster = data_arr[nonzero(cluster_assessment[:, 0].A == center)[0]]  # matrix.A返回矩阵的narray类型
            centroids[center, :] = mean(pts_in_cluster, axis=0)
    return centroids, cluster_assessment


def show_data(data_set, centroids, cluster_assessment):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data_arr = array(data_set)
    centroid_arr = array(centroids)

    data_arr = column_stack((data_arr, cluster_assessment[:, 0]))  # 扩展一列，表示该条数据属于某一簇
    ax.scatter(centroid_arr[:, 0], centroid_arr[:, 1], s=40, marker='+')
    # 这一句有点屌，根据最后一列簇类别来给每个数据点上色
    ax.scatter(data_arr[:, 0], data_arr[:, 1], c=15.0 * array(data_arr[:, -1]), marker='o')
    plt.show()


if __name__ == '__main__':
    d_m = load_data_set('./data/testSet.txt')
    centroids = random_centroids(d_m, 5)
    centroids, cluster_assessment = k_means(d_m, 4)
    show_data(d_m, centroids, cluster_assessment)


