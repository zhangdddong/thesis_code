#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# @license : Copyright(C), Your Company 
# @Author: Zhang Dong
# @Contact : 1010396971@qq.com
# @Date: 2020-08-11 17:25
# @Description: In User Settings Edit
# @Software : PyCharm
import numpy as np
import pandas as pd
import re
import utils
from configure import FLAGS
import OpenHowNet
import config
import jieba


def load_data_and_labels(path):
    hownet_dict = OpenHowNet.HowNetDict()
    data = []

    # 自定义用户词表
    my_user_word = set()
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            entity_head, entity_tail, relation, sentence = line.strip().split('\t')
            my_user_word.add(entity_head)
            my_user_word.add(entity_tail)
    for word in my_user_word:
        jieba.add_word(word)

    max_sentence_length = 0
    with open(path, encoding='UTF-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if len(line) == 0:
                continue
            entity_head, entity_tail, relation, sentence = line.split('\t')

            sentence = ' '.join(jieba.cut(sentence))
            sentence = sentence.replace(entity_head, ' e11 ' + entity_head + ' e12 ', 1)
            sentence = sentence.replace(entity_tail, ' e21 ' + entity_tail + ' e22 ', 1)

            tokens = sentence.split()
            if max_sentence_length < len(tokens):
                max_sentence_length = len(tokens)
            try:
                e1 = tokens.index("e12") - 1
            except Exception:
                print(i)
                continue
            if 'e22' not in sentence:
                len_head = len(entity_head)
                len_tail = len(entity_tail)
                e2 = e1 - (len_head - len_tail) if len_head > len_tail else e1 + (len_head - len_head)
            else:
                e2 = tokens.index("e22") - 1

            how_net_e1 = [entity_head] + list(hownet_dict.get_sememes_by_word(entity_head, lang='zh', structured=False, merge=True, expanded_layer=2))
            how_net_e2 = [entity_tail] + list(hownet_dict.get_sememes_by_word(entity_tail, lang='zh', structured=False, merge=True, expanded_layer=2))
            how_net = how_net_e1 + how_net_e2
            how_net = " ".join(how_net)
            data.append([i, sentence, e1, e2, relation, how_net])

    print(path)
    print("max sentence length = {}\n".format(max_sentence_length))

    df = pd.DataFrame(data=data, columns=["id", "sentence", "e1", "e2", "relation", "how_net"])

    pos1, pos2 = get_relative_position(df, FLAGS.max_sentence_length)

    df['label'] = [utils.class2label[r] for r in df['relation']]

    # Text Data
    x_text = df['sentence'].tolist()
    e1 = df['e1'].tolist()
    e2 = df['e2'].tolist()
    how_net = df['how_net'].tolist()

    # Label Data
    y = df['label']
    labels_flat = y.values.ravel()
    labels_count = np.unique(labels_flat).shape[0]

    # convert class labels from scalars to one-hot vectors
    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)

    return x_text, labels, e1, e2, pos1, pos2, how_net


def get_relative_position(df, max_sentence_length):
    # Position data
    pos1 = []
    pos2 = []
    for df_idx in range(len(df)):
        sentence = df.iloc[df_idx]['sentence']
        tokens = sentence.split()
        e1 = df.iloc[df_idx]['e1']
        e2 = df.iloc[df_idx]['e2']

        p1 = ""
        p2 = ""
        for word_idx in range(len(tokens)):
            p1 += str((max_sentence_length - 1) + word_idx - e1) + " "
            p2 += str((max_sentence_length - 1) + word_idx - e2) + " "
        pos1.append(p1)
        pos2.append(p2)

    return pos1, pos2


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == "__main__":
    trainFile = config.train_file
    testFile = config.test_file

    load_data_and_labels(testFile)
