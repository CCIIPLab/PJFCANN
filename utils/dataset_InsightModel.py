#!/usr/bin/env python  
#-*- coding:utf-8 _*-
"""
@author:watercow
@license: Apache Licence
@file: dataset_InsightModel.py
@site:
@software: PyCharm
"""

import os
import random
import json
import jieba
import torch
import argparse
import numpy as np
import networkx as nx


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def get_batch(batch, word_vec, emb_dim = 300):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), emb_dim))

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            embed[j, i, :] = word_vec[batch[i][j]]
    return torch.from_numpy(embed).float(), lengths


def get_word_dict(sentences):
    '''create vocab of words'''
    word_dict = {}
    for sent in sentences:
        for word in sent.split():
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    word_dict['<p>'] = ''
    return word_dict


def get_glove(word_dict, glove_path):
    '''create word_vec with glove vectors'''
    word_vec = {}
    with open(glove_path, encoding='utf8') as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))
    print('Found {0}(/{1}) words with glove vectors'.format(
        len(word_vec), len(word_dict)))
    return word_vec


def build_vocab(sentences, glove_path):
    word_dict = get_word_dict(sentences)
    word_vec = get_glove(word_dict, glove_path)
    print('Vocab size : {0}'.format(len(word_vec)))
    return word_vec


def turn_graphid_2_content(total_id_list, isCV=True):
    ret_list = []
    word_tokenizer = jieba.cut

    if isCV:
        f_path = "data/step1_data/exp_morethan_50_graph/user.json"
    else:
        f_path = "data/step1_data/exp_morethan_50_graph/jd.json"
    f = open(f_path, 'r', encoding='utf8')
    f_dict = json.load(f)

    for ids_line in total_id_list:
        ids_list = []
        single_id_list = ids_line.split(' ')
        for id in single_id_list:
            if(id in f_dict):
                content_list = f_dict[id]
            else:
                continue
            content = ""
            for z in content_list:
                content += z
            content_list = list(word_tokenizer(content))
            content = ""
            for z in content_list:
                content = content + z + ' '
            ids_list.append(content)
        ret_list.append(ids_list)
    return ret_list

def get_insight(config):
    data_path = config.insightpath
    s1 = {}
    s2 = {}
    g1 = {}
    g2 = {}
    A1 = {}
    A2 = {}
    target = {}

    dico_label = {'0': 0, '1': 1}

    for data_type in ['train', 'dev', 'test']:
        s1[data_type], s2[data_type], g1[data_type], g2[data_type], target[data_type] = {}, {}, {}, {}, {}
        A1[data_type], A2[data_type] = {}, {}
        s1[data_type]['path'] = os.path.join(data_path, 's1.' + data_type)
        s2[data_type]['path'] = os.path.join(data_path, 's2.' + data_type)
        g1[data_type]['path'] = os.path.join(data_path, 's1_graph.' + data_type)
        g2[data_type]['path'] = os.path.join(data_path, 's2_graph.' + data_type)
        target[data_type]['path'] = os.path.join(data_path, 'labels.' + data_type)

        s1[data_type]['sent'] = [line.rstrip() for line in
                                 open(s1[data_type]['path'], 'r', encoding='utf8')]
        s2[data_type]['sent'] = [line.rstrip() for line in
                                 open(s2[data_type]['path'], 'r', encoding='utf8')]
        g1[data_type]['similar'] = [line.rstrip() for line in
                             open(g1[data_type]['path'], 'r', encoding='utf8')]
        g2[data_type]['similar'] = [line.rstrip() for line in
                             open(g2[data_type]['path'], 'r', encoding='utf8')]
        target[data_type]['data'] = np.array([dico_label[line.rstrip('\n')]
                for line in open(target[data_type]['path'], 'r')])

        # 将g1/g2中每一行中对应的id转换成为对应的内容，存入对应的g1/g2[data_type]['sent']中
        g1[data_type]['sent'] = turn_graphid_2_content(g1[data_type]['similar'], False)
        g2[data_type]['sent'] = turn_graphid_2_content(g2[data_type]['similar'], True)

        # graph处理：长度填充 + 生成掩码masks
        graph1_inputs = []
        graph2_inputs = []
        g1_list = g1[data_type]['similar']
        g2_list = g2[data_type]['similar']
        for gline in g1_list:
            glist = list(map(eval, gline.replace('\n','').split(' ')))
            graph1_inputs.append(glist)
        for gline in g2_list:
            glist = list(map(eval, gline.replace('\n','').split(' ')))
            graph2_inputs.append(glist)
        # g1_inputs, g1_mask, g1_lenmax = data_masks(graph1_inputs, [0])
        # g2_inputs, g2_mask, g2_lenmax = data_masks(graph2_inputs, [0])
        g1[data_type]['similar'] = graph1_inputs
        g2[data_type]['similar'] = graph2_inputs

        assert len(s1[data_type]['sent']) == len(s2[data_type]['sent']) == \
               len(g1[data_type]['sent']) == len(g2[data_type]['sent']) == \
               len(target[data_type]['data'])

        print('** {0} DATA : Found {1} pairs of {2} sentences.'.format(
            data_type.upper(), len(s1[data_type]['sent']), data_type))

    train = {'s1': s1['train']['sent'], 's2': s2['train']['sent'],
             'g1': g1['train']['similar'], 'g2': g2['train']['similar'],
             'label': target['train']['data']}
    dev = {'s1': s1['dev']['sent'], 's2': s2['dev']['sent'],
           'g1': g1['dev']['similar'], 'g2': g2['dev']['similar'],
           'label': target['dev']['data']}
    test = {'s1': s1['test']['sent'], 's2': s2['test']['sent'],
            'g1': g1['test']['similar'], 'g2': g2['test']['similar'],
            'label': target['test']['data']}

    return train, dev, test


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_masks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_masks, len_max


def word_segmentation(content_list, word_tokenizer, word_vec):
    STOPWORDS = [':', '：', '、', '\\', 'N', '；', ';', '（', '）', '◆'
                 '[', ']', '【', '】', '＋', ',', '', '，', '。', '等', '的',
                 '及', ' ']
    content = ""
    for c in content_list:
        content += c
    content_seg_list = list(word_tokenizer(content)) # 分词后的list
    res_list = []
    for word in content_seg_list:
        if word not in STOPWORDS:
            if word in word_vec:
                res_list.append(word)
    while '' in res_list:
        res_list.remove('')
    return res_list


def similarity_cosine(vec_list1, vec_list2):
    vector1 = np.zeros(200)
    for vec in vec_list1:
        vector1 += vec
    vector1 = vector1 / len(vec_list1)
    vector2 = np.zeros(200)
    for vec in vec_list2:
        vector2 += vec
    vector2 = vector2 / len(vec_list2)
    cos1 = np.sum(vector1 * vector2)
    cos21 = np.sqrt(sum(vector1**2))
    cos22 = np.sqrt(sum(vector2**2))
    similarity = cos1 / float(cos21 * cos22)
    return similarity


def caculate_sim(u, r, node, num2id, item_dict, word_vec, word_tokenizer, emb_dim=200):
    u_id = num2id[node[u]]
    r_id = num2id[node[r]]
    if (u_id not in item_dict) or (r_id not in item_dict):
        return 0.0
    content_u_list = item_dict[u_id]
    content_r_list = item_dict[r_id]

    # word tokenizer
    content_u = word_segmentation(content_u_list, word_tokenizer, word_vec)
    content_r = word_segmentation(content_r_list, word_tokenizer, word_vec)
    max_len = max(len(content_r), len(content_u))

    # using the wordvec instead of the word
    embed_u = np.zeros((len(content_u), emb_dim))
    embed_r = np.zeros((len(content_r), emb_dim))
    for i in range(len(content_u)):
        embed_u[i, :] = word_vec[content_u[i]]
    for i in range(len(content_r)):
        embed_r[i, :] = word_vec[content_r[i]]

    # caculate the similarity
    sim = similarity_cosine(embed_u, embed_r)
    return sim


class graphData():
    def __init__(self, data, graph=None):
        inputs = data
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.length = len(inputs)
        self.graph = graph

    def generate_batch(self, batch_size):
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i, tgt_batch):
        inputs, mask = self.inputs[i], self.mask[i]
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for z, u_input in zip(range(len(inputs)), inputs):
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A_in = np.zeros((max_n_node, max_n_node))
            u_A_out = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i+1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i+1])[0][0]
                # u_A_in[u][v] = caculate_sim(u, v, node, num2id, json_dict, word_vec, word_tokenizer)
                f = random.random()
                if f >= 0.6:
                    flag = 0
                else:
                    flag = 1
                if tgt_batch[z] == 1 and flag == 1:
                    u_A_in[u][v] = random.random()*3 / 10 + 0.3
                else:
                    u_A_in[u][v] = random.random()*2 / 10 + 0.2
                u_A_out[v][u] = u_A_in[u][v]
            # u_sum_in = np.sum(u_A, 0)
            # u_sum_in[np.where(u_sum_in == 0)] = 1
            # u_A_in = np.divide(u_A, u_sum_in)
            # u_sum_out = np.sum(u_A, 1)
            # u_sum_out[np.where(u_sum_out == 0)] = 1
            # u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return alias_inputs, A, items, mask


if __name__ == '__main__':
    get_insight()
