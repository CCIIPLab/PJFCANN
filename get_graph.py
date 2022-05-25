#!/usr/bin/env python  
#-*- coding:utf-8 _*-
"""
@author:watercow
@license: Apache Licence
@file: get_graph.py
@site:
@software: PyCharm
"""
import os
import json
import jieba

word_tokenizer = jieba.cut

if __name__ == '__main__':
    graph_path = "data/train-test_data/s2_graph.train"
    cv_json_path = "data/step1_data/exp_morethan_50_graph/user.json"
    jd_json_path = "data/step1_data/exp_morethan_50_graph/jd.json"

    f_graph = open(graph_path, 'r')
    f_cv = open(cv_json_path, 'r', encoding='utf8')
    user_dict = json.load(f_cv)
    f_jd = open(jd_json_path, 'r', encoding='utf8')
    jd_dict = json.load(f_jd)

    for i in f_graph:
        cv_list = i.replace('\n', '').strip().split(' ')
        while '' in cv_list:
            cv_list.remove('')

        if(len(cv_list)):
            for j in cv_list:
                content_list = user_dict[j]

                content = ""
                for z in content_list:
                    content += z
                content_list = list(word_tokenizer(content))
                print(content_list)
