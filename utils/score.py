#!/usr/bin/env python  
#-*- coding:utf-8 _*-
"""
@author:watercow
@license: Apache Licence
@file: score.py
@site:
@software: PyCharm
"""
"""
Score the predictions with gold labels, using Accuracy, precision, recall, F1 and AUC metrics.
"""

import argparse
import torch
import numpy
import sys
from collections import Counter

class Scorer:
    def __init__(self):
        self.res = []
        self.gold = []
        self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0

    def add_batches_res(self, gold_res_list, model_res_list):
        batch_gold = gold_res_list
        batch_res = self._gen_res_list(model_res_list)
        if not isinstance(self.res, list):
            self.res = torch.cat((self.res, batch_res))
            self.gold = torch.cat((self.gold, batch_gold))
        else:
            self.res = batch_res
            self.gold = batch_gold

    def _gen_res_list(self, model_res_list):
        res = []
        for i in model_res_list:
            if abs(1-i) <= abs(i-0):
                res.append(1)
            else:
                res.append(0)
        return torch.from_numpy(numpy.array(res))

    def get_tp_tn_fp_fn(self):
        for i in range(len(self.gold)):
            if (self.gold[i] == 1 and self.res[i] == 1):
                self.tp += 1
            elif (self.gold[i] == 1 and self.res[i] == 0):
                self.fn += 1
            elif (self.gold[i] == 0 and self.res[i] == 1):
                self.fp += 1
            elif (self.gold[i] == 0 and self.res[i] == 0):
                self.tn += 1

    def Precision(self):
        if (self.tp + self.fp) == 0:
            return 0
        else:
            return self.tp / (self.tp + self.fp)

    def Recall(self):
        return self.tp / (self.tp + self.fn)

    def F1Score(self):
        return 2*((self.tp / (self.tp + self.fp)) * (self.tp / (self.tp + self.fn))) / \
               ((self.tp / (self.tp + self.fp)) + (self.tp / (self.tp + self.fn)))

    def Accuracy(self):
        return (self.tp+self.tn) / (self.tp + self.tn + self.fp +self.fn)

