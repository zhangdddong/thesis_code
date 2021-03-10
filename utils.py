#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# @license : Copyright(C), Your Company 
# @Author: Zhang Dong
# @Contact : 1010396971@qq.com
# @Date: 2020-08-11 15:55
# @Description: In User Settings Edit
# @Software : PyCharm
import numpy as np
import tensorflow as tf
import config


class2label = dict()
label2class = dict()
with open(config.relation_file, encoding='UTF-8') as f:
    for line in f:
        pairs = line.strip().split()
        class2label[pairs[0]] = int(pairs[1])
        label2class[int(pairs[1])] = pairs[0]


def initializer():
    return tf.keras.initializers.glorot_normal()


def load_word2vec(word2vec_path, embedding_dim, vocab):
    vocab_len = len(vocab.vocabulary_)
    init_w = np.random.randn(vocab_len, embedding_dim).astype(np.float32) * np.sqrt(2.0 / vocab_len)
    print('Load word2vec file {0}'.format(word2vec_path))
    with open(word2vec_path, 'rb') as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('latin-1')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            idx = vocab.vocabulary_.get(word)
            if idx != 0:
                init_w[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return init_w


def load_glove(glove_path, embedding_dim, vocab):
    init_w = np.random.randn(len(vocab.vocabulary_), embedding_dim).astype(np.float32) * np.sqrt(
        2.0 / len(vocab.vocabulary_))
    print("Load glove file {0}".format(glove_path))
    f = open(glove_path, 'r', encoding='utf8')
    for line in f:
        split_line = line.split(' ')
        word = split_line[0]
        embedding = np.asarray(split_line[1:], dtype='float32')
        idx = vocab.vocabulary_.get(word)
        if idx != 0:
            init_w[idx] = embedding
    return init_w
