from math import log
import operator


def create_data_set():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return data_set, labels


def calculate_entropy(data_set):
    """
        step1:统计每个label出现的次数
        step2:计算每一个label的熵，然后相加
    :param data_set:
    :return:
    """
    num_entries = len(data_set)
    label_counts = {}
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    entropy = 0.0
    for key in label_counts.keys():
        prob = float(label_counts[key]) / num_entries  # 每个label出现的概率
        entropy -= prob * log(prob, 2)
    return entropy


def split_data_set(data_set, axis, value):
    """
        划分数据集。
        在axis这一特征(列上)=value，就保留该该数据，并删除axis这一列
    :param data_set: 待划分的数据集
    :param axis: 划分数据集的[特征属性]
    :param value: 需要返回的特征的值
    :return: 在axis轴上符合value值的数据集合
    """
    new_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            new_data_set.append(reduced_feat_vec)
    return new_data_set


def choose_best_feature_to_split(data_set):
    """
        选择最好的数据集划分方式。
        对特征属性列进行循环操作，计算其信息增益，再跟原始熵做比较
        如：[[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
        step1:找寻特征属性个数，这里是第一列和第二列；计算原始熵
        step2:遍历特征属性，如遍历到第一列时
        step3:第一列特征属性值有两种：[1，0]，遍历这两种特征属性值，如遍历到特征属性值为1时；
        step4:当遍历到第一个特征属性且为第一个特征属性值时，(利用这两个参数)划分原始数据集，并度量划分数据集的熵
        ----------收尾------------
        step5:[1, 0]特征属性值计算出的熵之和与原始熵比较，小就确认该列特征属性为最好的数据集划分方式
    :param data_set:
    :return:
    """
    num_features = len(data_set[0]) - 1  # 最后一列是label，不能算是[特征属性]
    base_entropy = calculate_entropy(data_set)  # 计算整个数据集的原始熵（用的label标签作为特征属性）
    best_info_gain = 0.0  # 信息增益，即熵的减少或数据无序度减少
    best_feature = -1  # 最好划分的特征属性
    for i in range(num_features):
        feature_list = [example[i] for example in data_set]
        unique_values = set(feature_list)  # 去重
        new_entropy = 0.0
        for value in unique_values:  # 对每一个特征划分一次数据集
            sub_data_set = split_data_set(data_set, i, value)  # 挑选出数据集 在特征属性i列中 = value(需要返回的特征的值)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calculate_entropy(sub_data_set)  # 每个特征属性占整体的比例 *
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    """
        划分数据集时的数据路径。
        作用：将class_list中出现频率最该的那个类标签筛选出来
        如：['yes', 'yes', 'no', 'no', 'no']，则筛选出：no
    :param class_list:
    :return:
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


# 创建树的递归函数代码
def create_tree(data_set, labels):
    """
        递归停止条件：
        1.所有类标签完全相同，返回当前类标签
        2.使用完所有特征，仍然不能将数据集划分成仅包含唯一类别的分组

    :param data_set:
    :param labels: labels = ['no surfacing', 'flippers']
    :return:
    """
    class_list = [example[-1] for example in data_set]  # 'yes''no' data_set的最后一列为类标签
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) == 1:  # 只有一个特征属性，即只有一列
        return majority_cnt(class_list)
    best_feat = choose_best_feature_to_split(data_set)  # 选择最好的数据集划分方式。返回值为某一特征属性列的索引
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    del(labels[best_feat])
    feat_values = [example[best_feat] for example in data_set]
    unique_values = set(feat_values)
    for value in unique_values:
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return my_tree


#
def classify(input_tree, feat_labels, test_vec):
    first_str = input_tree.keys()[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    key = test_vec[feat_index]
    value_of_feat = second_dict[key]
    if isinstance(value_of_feat, dict):
        class_label = classify(value_of_feat, feat_labels, test_vec)
    else:
        class_label = value_of_feat
    return class_label


def store_tree(input_tree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(input_tree, fw)
    fw.close()


def grab_tree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

if __name__ == '__main__':
    my_data, labels = create_data_set()
    print(create_tree(my_data, labels))

