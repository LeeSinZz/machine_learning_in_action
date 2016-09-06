from numpy import array, mat, zeros
import matplotlib.pylab as plt
"""
    关联分析是一种在大规模数据集中寻找有趣关系的任务。
[关系]有两种形式：
1.频繁项集(frequent item sets)：经常出现在一块的物品的集合。
2.关联规则(association rules)：暗示两种物品之间可能存在很强的关系。
    支持度：数据集中包含该项集的记录所占的比例。如交易记录中有多少条记录包含尿布。“count({尿布}) / 总交易记录”
    可信度：针对一条如{尿布}→{葡萄酒}的关联规则来定义的。如“支持度({尿布，葡萄酒}) / 支持度({尿布})”

apriori
step1: 生成候选项集；
step2: 通过最小支持率筛选候选项集及并得到其支持率
step3:

"""


def load_data_set():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def create_c1(data_set):
    """
        构建候选项集的集合。每一个候选项集元素只有1个。如：[[1], [2], [3], [4], [5]]
    :param data_set: 数据集
    :return:
    """
    candidates_1 = []
    for transaction in data_set:
        for item in transaction:
            if [item] not in candidates_1:
                candidates_1.append([item])
        candidates_1.sort()
    return list(map(frozenset, candidates_1))


def scan_data_set(data_set, candidates_k, min_support_rate):
    """
        满足最小支持率的候选项集的集合。
    :param data_set: 数据集
    :param candidates_k: 候选项集的集合，每个候选项集元素为k个
    :param min_support_rate: 最小支持率
    :return:
    """
    record_count = {}
    for transaction in data_set:  # 每一笔交易
        for candidate in candidates_k:  # 每一个候选项集
            if candidate.issubset(transaction):
                if candidate not in record_count.keys():  # 判断dict是否含有key，用 ** in dict 语法
                    record_count[candidate] = 1
                else:
                    record_count[candidate] += 1
    transaction_count = len(data_set)
    satisfy_min_support_rate_collections = []  # 大于最小支持率的候选项集
    all_support_rate_data = {}  # 所有候选项集及其支持率
    for key in record_count.keys():
        single_support_rate = record_count.get(key) / transaction_count
        if single_support_rate >= min_support_rate:
            satisfy_min_support_rate_collections.insert(0, key)  # 数据插入list的首位
        all_support_rate_data[key] = single_support_rate
    return satisfy_min_support_rate_collections, all_support_rate_data


def create_collection_k(collections_k, k):
    satisfy_min_support_rate_collections = []
    collections_length = len(collections_k)
    for i in range(collections_length):
        for j in range(i+1, collections_length):
            collections_1 = list(collections_k[i])[:k-2]
            collections_2 = list(collections_k[j])[:k-2]
            collections_1.sort()
            collections_2.sort()
            if collections_1 == collections_2:
                satisfy_min_support_rate_collections.append(collections_k[i] | collections_k[j])  # | 两个集合的联合(并)操作
    return satisfy_min_support_rate_collections


def apriori(data_set, min_support_rate=0.5):
    collections_1 = create_c1(data_set)  # collections_1和下面的L1是一个意思
    unique_data_set = list(map(set, data_set))
    L1, support_rate_data = scan_data_set(unique_data_set, collections_1, min_support_rate)
    L = [L1]
    k = 2
    while len(L[k-2]) > 0:
        collections_k = create_collection_k(L[k-2], k)
        Lk, support_k = scan_data_set(unique_data_set, collections_k, min_support_rate)
        support_rate_data.update(support_k)
        L.append(Lk)
        k += 1
    return L, support_rate_data


def rule_from_collection(collection, H, support_rate_data, big_rule_list, min_confidence_value):
    m = len(H[0])
    if len(collection) > (m + 1):
        Hmp1 = apriori(H, m+1)
        Hmp1 = calculate_config(collection, Hmp1, support_rate_data, min_confidence_value)
        if len(Hmp1) > 1:
            rule_from_collection(collection, Hmp1, support_rate_data, min_confidence_value)


def calculate_config(collection, H, support_rate_data, min_confidence_value):
    pruned_h = []
    for conseq in H:
        conf = support_rate_data[collection] / support_rate_data[collection - conseq]
        if conf >= min_confidence_value:
            support_rate_data.append((collection-conseq), '-->', conseq, 'conf:', conf)
            pruned_h.append(conseq)
    return pruned_h


def generate_rules(L, support_rate_data, min_confidence_value):
    big_rule_list = []
    for i in range(1, len(L)):
        for collection in L[i]:
            H1 = [frozenset([item]) for item in frozenset]
            if i > 1:
                rule_from_collection(collection, H1, support_rate_data, big_rule_list, min_confidence_value)
            else:
                calculate_config(collection, H1, support_rate_data, min_confidence_value)
    return big_rule_list


if __name__ == '__main__':
    d_s = load_data_set()
    L, support_rate_data = apriori(d_s)
    print(L)