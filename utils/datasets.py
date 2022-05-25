#!/usr/bin/env python  
#-*- coding:utf-8 _*-
"""
@author:watercow
@license: Apache Licence
@file: datasets.py
@site:
@software: PyCharm
"""

import numpy as np
import json
import random
import torch
from tqdm import tqdm
from torch.autograd import Variable

# some global var
NULL = "-NULL-"
UNK = "-UNK-"

def _make_word_vector(exp_list, w2i, seq_len, seg_len=15):
    '''
    This function completes the function of filling sentences with different lengths and different pq
    '''
    index_vec_list = []
    for exp in exp_list:
        index_vec = [w2i[w] if w in w2i else w2i[UNK] for w in exp]
        pad_len = max(0, seq_len - len(index_vec))
        index_vec += [w2i[NULL]] * pad_len
        index_vec = index_vec[:seq_len]
        index_vec_list.append(index_vec)

    while (len(index_vec_list) < seg_len):
        index_vec = [w2i[NULL]] * seq_len
        index_vec_list.append(index_vec)

    return index_vec_list

def make_vector(batch, w2i, j_sent_len, r_sent_len):
    '''
    encoding with padding
    '''
    j, r = [], []
    # batch = (j, r, label)
    for d in batch:
        j.append(_make_word_vector(d[0], w2i, j_sent_len))
        r.append(_make_word_vector(d[1], w2i, r_sent_len))

    j = to_var(torch.LongTensor(j))
    r = to_var(torch.LongTensor(r))

    return j, r

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def load_processed_json(fpath_data, fpath_shared):
    # shared -----------
    # word_counter: word and how many times it appeared
    # word2vec: word2vec pretrained weights
    # data -----------
    # job_posting: job posting saving in a list
    # resume: candidate resume saving in a list
    # label: indicate whether this pair is finally matched
    # job_id_list: job posting id saving in a list
    # resume_id_list: resume id saving in a list
    # pair_id_list

    data = json.load(open(fpath_data, encoding='utf8'))
    shared = json.load(open(fpath_shared, encoding='utf8'))
    return data, shared

def load_glove_weights(glove_path, embd_dim, vocab_size, word_index):
    embeddings_index = {}
    with open(glove_path, encoding='utf8') as f:
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            try:
                vector = np.array(values[1:], dtype='float32')
                embeddings_index[word] = vector
            except:
                continue

    print('Found %s word-vectors in GLOVE file.' % len(embeddings_index))
    embedding_matrix = np.zeros((vocab_size, embd_dim))
    print('embed_matrix.shape', embedding_matrix.shape)
    found_cnt = 0
    total_word_cnt = 0

    for word, i in word_index.items():
        total_word_cnt += 1
        embedding_vector = embeddings_index.get(word)
        # words not found in embedding index will be all-zeros
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            found_cnt += 1

    print(found_cnt, 'words are found in GLOVE file')
    print('{:.3f}% words are found in GLOVE'.format((found_cnt / total_word_cnt) * 100))
    return embedding_matrix

class Dataset:
    def __init__(self, data, shared):
        self.shared = shared
        self.q_num = 15
        self.r_max_len = 300
        self.j_max_len = 30
        self.data = self.init_data(data)

    def init_data(self, data):
        job_posting = data['job_posting']
        resume = data['resume']

        for i, job in enumerate(job_posting):
            q = len(job)
            if q > self.q_num:
                random.shuffle(job)
                job_posting[i] = job[0 : self.q_num-1]

        for i, job in enumerate(job_posting):
            for j, req in enumerate(job):
                if len(req) > self.j_max_len:
                    job_posting[i][j] = req[0 : self.j_max_len-1]

        for i, cv in enumerate(resume):
            for j, exp in enumerate(cv):
                if len(exp) > self.r_max_len:
                    resume[i][j] = exp[0 : self.r_max_len-1]

        data['job_posting'] = job_posting
        data['resume'] = resume

        return data

    def size(self):
        '''
        :return: how many pairs(j-r)
        '''
        return len(self.data['pair_id_list'])

    def get_sent_maxlen(self):
        # word level content max length
        job_max_len = max([len(posting) for job in self.data['job_posting'] for posting in job])
        resume_max_len = max([len(ability) for exp in self.data['resume'] for ability in exp])
        return job_max_len, resume_max_len

    def get_p_and_q_maxlen(self):
        # p denotes the number of exp
        # q denotes the number of req
        p_max_len = max([len(cv) for cv in self.data['resume']])
        q_max_len = max([len(job) for job in self.data['job_posting']])
        return q_max_len, p_max_len

    def get_batches(self, batch_size, shuffle=False):
        batches = []
        batch = []

        for i in range(self.size()):
            j = self.data['job_posting'][i]
            r = self.data['resume'][i]
            label = self.data['label'][i]

            batch.append((j, r, label))

            if len(batch) == batch_size:
                batches.append(batch)
                batch = []

        if shuffle:
            random.shuffle(batches)
        return batches

    def get_word_index(self, word_count_th=10):
        word2vec_dict = self.get_word2vec()
        word_counter = self.get_word_counter()

        w2i = {
            w : i for i, w in enumerate(
                w for w, ct in word_counter.items()
                if ct > word_count_th or (w in word2vec_dict)
            )
        }
        return w2i

    def get_word2vec(self):
        return self.shared['word2vec']

    def get_word_counter(self):
        return self.shared['word_counter']