#!/usr/bin/env python  
#-*- coding:utf-8 _*-
"""
@author:watercow
@license: Apache Licence
@file: InsightModel.py
@site:
@software: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import jieba
import json
from model.GraphModel import *
from model.SentSimilarityModel import *


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def load_num2id(config):
    f_save_unum2id = open(config['unum2id'], 'r', encoding='utf8')
    unum2id = json.load(f_save_unum2id)
    f_save_jnum2id = open(config['jnum2id'], 'r', encoding='utf8')
    jnum2id = json.load(f_save_jnum2id)
    f_save_unum2id.close()
    f_save_jnum2id.close()
    return unum2id, jnum2id


def load_jd_user(config):
    f_jd = open(config['jd_json'], 'r', encoding='utf8')
    jd_dict = json.load(f_jd)
    f_user = open(config['user_json'], 'r', encoding='utf8')
    user_dict = json.load(f_user)
    return jd_dict, user_dict


class InsightModel(nn.Module):
    def __init__(self, config, word_vec):
        super(InsightModel, self).__init__()
        # ======================
        #   Text Matching Part
        # ======================
        self.encoder = SentSimilarityModel(config)
        # ======================
        #        GNN Part
        # ======================
        self.gnn_J = SessionGraph(config, config['n_J_node'])
        self.gnn_R = SessionGraph(config, config['n_R_node'])
        self.jd_dict, self.user_dict = load_jd_user(config)
        self.word_vec = word_vec
        self.word_tokenizer = jieba.cut
        self.unum2id, self.jnum2id = load_num2id(config)
        # ======================
        #       Classifier
        # ======================
        self.nonlinear_fc = config['nonlinear_fc']
        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.encoder_type = config['encoder_type']
        self.dpout_fc = config['dpout_fc']
        self.inputdim = self.encoder.inputdim + 4*self.gnn_J.hidden_size

        if config['nonlinear_fc']:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.n_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.n_classes)
            )

    def forward(self, s1, s2, g1, g2, slice, tgt_batch):
        # Local Information
        H_local_R, H_local_J = self.encoder(s1, s2)

        # Global Information
        # prepare Graph-match data batch
        alias_inputs1, A1, items1, mask1 = g1.get_slice(slice, tgt_batch)
        alias_inputs2, A2, items2, mask2 = g2.get_slice(slice, tgt_batch)
        alias_inputs1 = trans_to_cuda(torch.Tensor(alias_inputs1).long())
        alias_inputs2 = trans_to_cuda(torch.Tensor(alias_inputs2).long())
        items1 = trans_to_cuda(torch.Tensor(items1).long())
        items2 = trans_to_cuda(torch.Tensor(items2).long())
        A1 = trans_to_cuda(torch.Tensor(A1).float())
        A2 = trans_to_cuda(torch.Tensor(A2).float())
        mask1 = trans_to_cuda(torch.Tensor(mask1).long())
        mask2 = trans_to_cuda(torch.Tensor(mask2).long())

        hidden_J = self.gnn_J(items1, A1)
        hidden_R = self.gnn_R(items2, A2)

        getJ = lambda i: hidden_J[i][alias_inputs1[i]]
        getR = lambda i: hidden_R[i][alias_inputs2[i]]
        length1 = torch.arange(len(alias_inputs1))
        length2 = torch.arange(len(alias_inputs2))
        seq_hidden1 = torch.stack([getJ(i) for i in torch.arange(len(alias_inputs1)).long()])
        seq_hidden2 = torch.stack([getR(i) for i in torch.arange(len(alias_inputs2)).long()])
        H_global_J = self.gnn_J.compute_globalhidden(seq_hidden1, mask1)
        H_global_R = self.gnn_R.compute_globalhidden(seq_hidden2, mask2)

        # Concate
        H_J = torch.cat([H_local_J, H_global_J], 1)
        H_R = torch.cat([H_local_R, H_global_R], 1)

        # Classifer
        # features = torch.cat((H_local_J, H_local_R, torch.abs(H_local_J - H_local_R), H_local_J*H_local_R), 1)
        features = torch.cat((H_J, H_R, torch.abs(H_J - H_R), H_J*H_R), 1)
        output = self.classifier(features)
        return output

