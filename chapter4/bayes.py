#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy import array, log, ones, random
import re


def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return posting_list, class_vec


def create_vocab_list(data_set):
    """
        创建一个包含所有文档中出现的不重复词的列表。
    :param data_set:
    :return:
    """
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def set_of_words_to_vec(vocab_list, input_set):
    """
        词表到向量的转换函数。（词集模型set-of-words model:每个词出现与否作为一个特征）
        新建一个和不重复词列表长度相同且值为0的向量列表，遍历input_set，
        如果有词在vocab_list中出现就将向量列表该位置设置成1
    :param vocab_list:不重复词列表
    :param input_set:某个文档或某句话
    :return:词向量,如[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    """
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return return_vec


def bag_of_words_to_vec(vocab_list, input_set):
    """
        词袋模型bag-of-words model:会记录每个单词出现的次数
    :param vocab_list:
    :param input_set:
    :return:
    """
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec


def train_nbo(train_matrix, train_category):
    """
        朴素贝叶斯分类器训练函数。

    :param train_matrix:文档矩阵
    :param train_category:每篇文档类别标签构成的向量
    :return:
    """
    num_train_docs = len(train_matrix)  # 文章数
    num_words = len(train_matrix[0])  # 每篇文章单词数
    prob_abusive = sum(train_category)/float(num_train_docs)  # 含有辱骂性词语的文章占的比例
    # 初始化概率.denom分母项，abusive辱骂
    prob0_num = ones(num_words)
    prob1_num = ones(num_words)
    prob0_denom = 2.0
    prob1_denom = 2.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            # 辱骂性文章 向量相加
            prob1_num += train_matrix[i]  # prob1_num长度为32的别表，相加指每个位置上的数字相加
            prob1_denom += sum(train_matrix[i])
        else:
            prob0_num += train_matrix[i]
            prob0_denom += sum(train_matrix[i])
    prob1_vector = log(prob1_num/prob1_denom)  # 使用log()函数防止乘积太小而引起程序下溢出
    prob0_vector = log(prob0_num/prob0_denom)
    return prob0_vector, prob1_vector, prob_abusive


def classify_nb(vec_to_classify, prob0_vector, prob1_vector, prob1_class):
    """
        朴素贝叶斯分类函数。
    :param vec_to_classify:要分类的向量，经过set_of_words_to_vec(my_vocab_list, test_entry)函数处理过的
    :param prob0_vector:没辱骂性词语文章类别的向量
    :param prob1_vector:有辱骂性词语文章类别的向量
    :param prob1_class:有辱骂性词语文章类别的概率
    :return:分类结果
    """
    p1 = sum(vec_to_classify * prob1_vector) + log(prob1_class)  # 向量相乘就是每一项相乘得出一个新向量
    p0 = sum(vec_to_classify * prob0_vector) + log(1 - prob1_class)
    if p1 > p0:
        return 1
    else:
        return 0


def testing_nb():
    """
        便利函数(convenience function),封装了所有操作
    :return:
    """
    list0_posts, list_classes = load_data_set()
    my_vocab_list = create_vocab_list(list0_posts)
    train_matrix = []
    for doc in list0_posts:
        train_matrix.append(set_of_words_to_vec(my_vocab_list, doc))
    prob0_vector, prob1_vector, prob_abusive = train_nbo(train_matrix, list_classes)
    test_entry = ['love', 'my', 'dalmation']
    this_doc = array(set_of_words_to_vec(my_vocab_list, test_entry))
    print(test_entry, ' classified as: ', classify_nb(this_doc, prob0_vector, prob1_vector, prob_abusive))
    test_entry = ['stupid', 'garbage']
    this_doc = array(set_of_words_to_vec(my_vocab_list, test_entry))
    print(test_entry, ' classified as: ', classify_nb(this_doc, prob0_vector, prob1_vector, prob_abusive))


# ++++++++++测试算法：使用朴素贝叶斯进行交叉验证++++++++++++
def text_parse(big_string):
    list_of_tokens = re.split(r'\W*', big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok)>2]


def spam_test():
    doc_list = []
    class_list = []
    full_text = []
    for i in range(1, 26):
        word_list = text_parse(open('email/spam/%d.txt' % i, 'r').read())
        print(word_list)
        doc_list.append(word_list)
        full_text.append(word_list)
        class_list.append(1)
        word_list = text_parse(open('email/ham/%d.txt' % i, 'r').read())
        doc_list.append(word_list)
        full_text.append(word_list)
        class_list.append(0)
    vocab_list = create_vocab_list(doc_list)

    training_set = list(range(50))
    test_set = []
    for i in range(10):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])
    train_matrix = []
    train_classes = []
    for doc_index in training_set:
        train_matrix.append(set_of_words_to_vec(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    prob0_vector, prob1_vector, prob_spam = train_nbo(train_matrix, train_classes)
    error_count = 0
    for doc_index in test_set:
        word_vector = set_of_words_to_vec(vocab_list, doc_list[doc_index])
        if classify_nb(array(word_vector), prob0_vector, prob1_vector, prob_spam) != class_list[doc_index]:
            error_count += 1
    print('the error rate is: ', float(error_count)/len(test_set))


if __name__ == '__main__':
    spam_test()









