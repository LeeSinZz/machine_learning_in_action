import matplotlib.pyplot as plt
from numpy import array, mat, zeros, ones, matrix, shape, inf, log, multiply, exp, sign


def load_data_set(file_name):
    # data_mat = matrix([[1.0, 2.1], [2.0, 1.1], [1.3, 1.0], [1.0, 1.0], [2.0, 1.0]])
    # class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    # return data_mat, class_labels
    with open(file_name, 'r') as fr:
        feature_num = len(fr.readline().split('\t'))
        data_mat = []
        label_mat = []
        for line in fr.readlines():
            line_arr = []
            current_line = line.strip().split('\t')
            for i in range(feature_num-1):
                line_arr.append(float(current_line[i]))
            data_mat.append(line_arr)
            label_mat.append(float(current_line[-1]))
    return data_mat, label_mat


def stump_classify(data_mat, dimension, threshold_value, threshold_inequality):
    """
        通过与阈值的比较对数据进行分类
    :param data_mat: 训练数据
    :param dimension: 对哪一维(feature)的数据进行分类
    :param threshold_value: 阈值
    :param threshold_inequality: 分类符号。有点绕，自己画图体会。
    :return:
    """
    return_array = ones((shape(data_mat)[0], 1))
    if threshold_inequality == 'lt':
        return_array[data_mat[:, dimension] <= threshold_value] = -1.0
    else:
        return_array[data_mat[:, dimension] > threshold_value] = -1.0
    return return_array


def build_stump(data_arr, class_labels, D):
    data_mat = mat(data_arr)
    label_mat = mat(class_labels).T
    r, c = shape(data_mat)
    number_steps = 10.0
    best_stump = {}
    best_class_estimate = mat(zeros((r, 1)))
    min_error = inf  # 正无穷
    for i in range(c):
        range_min = data_mat[:, 1].min()
        range_max = data_mat[:, 1].max()
        step_size = (range_max - range_min) / number_steps
        for j in range(-1, int(number_steps) + 1):
            for inequality in ['lt', 'gt']:
                threshold_value = (range_min + float(j) * step_size)
                predicted_values = stump_classify(data_mat, i, threshold_value, inequality)
                error_arr = mat(ones((r, 1)))
                error_arr[predicted_values == label_mat] = 0
                weighted_error = D.T * error_arr
                # print('split: dimension %d, threshold % .2f, threshold inequality : %s, weighted error %.3f' % (i, threshold_value, inequality, weighted_error))
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_class_estimate = predicted_values.copy()
                    best_stump['dime'] = i
                    best_stump['thresh'] = threshold_value
                    best_stump['ineq'] = inequality
    return best_stump, min_error, best_class_estimate


def adaptive_boost_ds(data_arr, class_labels, weak_classify_num=50):
    weak_classifier_arr = []
    r = shape(data_arr)[0]
    D = mat(ones((r, 1)) / r)  # D是一个概率分布向量，所以元素之和为0
    aggregation_class_estimate = mat(zeros((r, 1)))  # 记录每个数据点的类别估计累计值
    total_error_rate = 0.0
    for i in range(weak_classify_num):
        best_stump, error, class_estimate = build_stump(data_arr, class_labels, D)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))  # 1e-16 防止除0溢出
        best_stump['alpha'] = alpha
        weak_classifier_arr.append(best_stump)
        expon = multiply(-1 * alpha * mat(class_labels).T, class_estimate)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        aggregation_class_estimate += alpha * class_estimate
        # print('aggregation_class_estimate: ', aggregation_class_estimate.T)
        aggregation_errors = multiply(sign(aggregation_class_estimate) != mat(class_labels).T, ones((r, 1)))
        error_rate = aggregation_errors.sum() / r
        total_error_rate += error_rate
        if error_rate == 0.0:
            break
    print('++++训练错误率: ', total_error_rate / weak_classify_num)
    return weak_classifier_arr


def test_adaptive_classify(data_to_test, classifier_arr):
    data_mat = mat(data_to_test)
    r = shape(data_mat)[0]
    aggregation_class_estimate = mat(zeros((r, 1)))
    for i in range(len(classifier_arr)):
        class_estimate = stump_classify(data_mat, int(classifier_arr[i]['dime']), classifier_arr[i]['thresh'], classifier_arr[i]['ineq'])
        aggregation_class_estimate += classifier_arr[i]['alpha'] * class_estimate
    return sign(aggregation_class_estimate)


if __name__ == '__main__':
    d_m, c_m = load_data_set('./data/horseColicTraining2.txt')
    weak_classifier_arr = adaptive_boost_ds(d_m, c_m)

    d_test, c_test = load_data_set('./data/horseColicTest2.txt')
    predict_result = test_adaptive_classify(d_test, weak_classifier_arr)
    r, c = shape(d_test)
    error_arr = mat(ones((r, 1)))
    c_test = mat(c_test).T
    error_arr[predict_result == c_test] = 0
    error_num = 0
    for i in range(r):
        if error_arr[i, :] == 1:
            error_num += 1
    print('error number: %d' % error_num)
    print('error rate: %.2f' % (error_num / r, ))
























