#!/usr/bin/env python  
#-*- coding:utf-8 _*-
"""
@author:watercow
@license: Apache Licence
@file: run_InsightModel.py
@site:
@software: PyCharm
"""

import os
import sys
import time
import pickle
import argparse

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.mutils import get_optimizer
from model.InsightModel import InsightModel
from utils.dataset_InsightModel import graphData, trans_to_cuda
from utils.dataset_InsightModel import get_insight, get_batch, build_vocab

parser = argparse.ArgumentParser(description='Insight Training')
# ========
# path
# ========
parser.add_argument("--insightpath", default='data/train-test_data/', help="Insight data path")
parser.add_argument("--word_emb_path", default='data/embedding/Tencent_AILab_ChineseEmbedding.txt')
parser.add_argument("--word_emb_dim", type=int, default=200, help="word embedding dimension")
parser.add_argument("--outputdir", type=str, default='checkpoint/Insight', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='model.pickle')
parser.add_argument("--unum2id", default="data/step1_data/unum2id.json")
parser.add_argument("--jnum2id", default="data/step1_data/jnum2id.json")
parser.add_argument("--jd_json", default="data/step1_data/exp_morethan_50_graph/jd.json")
parser.add_argument("--user_json", default="data/step1_data/exp_morethan_50_graph/user.json")

# ========
# training
# ========
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc")
parser.add_argument("--optimizer", default='sgd,lr=0.01', help='adam or sgd,lr=0.1')
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

# =======
# model -- LSTM
# =======
parser.add_argument("--encoder_type", type=str, default='InferSent', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=2, help="0/1")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")

# =======
# model -- GNN
# =======
parser.add_argument("--n_J_node", default=264565)
parser.add_argument("--n_R_node", default=4465)
parser.add_argument("--step", default=1, help='GNN propogation steps')
parser.add_argument("--hiddenSize", default=200, help='hidden state size of gnn')
parser.add_argument("--nonhybrid", action='store_true', help='only use the global preference to predict')

# ========
# gpu
# ========
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")

params, _ = parser.parse_known_args()

# set gpu device
torch.cuda.set_device(params.gpu_id)

"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)

"""
DATA
"""
train, valid, test = get_insight(params)
word_vec = build_vocab(train['s1'] + train['s2'] +
                       valid['s1'] + valid['s2'] +
                       test['s1'] + test['s2'], params.word_emb_path)
for split in ['s1', 's2']:
    for data_type in ['train', 'valid', 'test']:
        eval(data_type)[split] = np.array([['<s>'] +
            [word for word in sent.split() if word in word_vec] +
            ['</s>'] for sent in eval(data_type)[split]])
for split in ['g1', 'g2']:
    for data_type in ['train', 'valid', 'test']:
        eval(data_type)[split] = np.array([session
            for session in eval(data_type)[split]])

"""
MODEL
"""
# model config
config_Insight_model = {
    'n_words'       :   len(word_vec)       ,
    'word_emb_dim'  :   params.word_emb_dim ,
    'enc_lstm_dim'  :   params.enc_lstm_dim ,
    'n_enc_layers'  :   params.n_enc_layers ,
    'dpout_model'   :   params.dpout_model  ,
    'dpout_fc'      :   params.dpout_fc     ,
    'fc_dim'        :   params.fc_dim       ,
    'bsize'         :   params.batch_size   ,
    'n_classes'     :   params.n_classes    ,
    'pool_type'     :   params.pool_type    ,
    'nonlinear_fc'  :   params.nonlinear_fc ,
    'encoder_type'  :   params.encoder_type ,
    'use_cuda'      :   True                ,
    'n_J_node'      :   params.n_J_node     ,
    'n_R_node'      :   params.n_R_node     ,
    'nonhybrid'     :   params.nonhybrid    ,
    'hiddenSize'    :   params.hiddenSize   ,
    'step'          :   params.step         ,
    'jd_json'       :   params.jd_json      ,
    'user_json'     :   params.user_json    ,
    'unum2id'       :   params.unum2id      ,
    'jnum2id'       :   params.jnum2id      ,
}

# model
encoder_types = ['InferSent', 'BLSTMprojEncoder', 'BGRUlastEncoder',
                 'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                str(encoder_types)

Insight_model = InsightModel(config_Insight_model, word_vec)
print(Insight_model)

# loss
weight = torch.FloatTensor(params.n_classes).fill_(1)
loss_fn = nn.CrossEntropyLoss(weight=weight)
loss_fn.size_average = False

# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
optimizer = optim_fn(Insight_model.parameters(), **optim_params)

# cuda by default
Insight_model.cuda()
loss_fn.cuda()

"""
TRAIN
"""
val_acc_best = -1e10
adam_stop = False
stop_training = False
lr = optim_params['lr'] if 'sgd' in params.optimizer else None


def trainepoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    Insight_model.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    # shuffle the data
    permutation = np.random.permutation(len(train['s1']))

    s1 = train['s1'][permutation]
    s2 = train['s2'][permutation]
    g1 = train['g1'][permutation]
    g2 = train['g2'][permutation]
    target = train['label'][permutation]

    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch > 1\
        and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
    print('Learning Rate : {0}'.format(optimizer.param_groups[0]['lr']))

    g1 = graphData(g1)
    g2 = graphData(g2)
    slices = g1.generate_batch(params.batch_size)

    for slice, stidx in zip(slices, range(0, len(s1), params.batch_size)):
        # prepare Inner-match data batch
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size],
                                     word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size],
                                     word_vec, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size])).cuda()
        k = s1_batch.size(1) # actual batch_size

        # model forward
        output = Insight_model((s1_batch, s1_len), (s2_batch, s2_len), g1, g2, slice, tgt_batch)
        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
        assert len(pred) == len(s1[stidx:stidx + params.batch_size])

        # loss
        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.item())
        words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping (default: OFF)
        shrink_factor = 1
        total_norm = 0

        # for p in Insight_model.parameters():
        #     if p.requires_grad:
        #         p.grad.data.div_(k) # divide by the actual batch size
        #         total_norm += p.grad.data.norm() ** 2
        # total_norm = np.sqrt(total_norm.cpu())

        if total_norm > params.max_norm:
            shrink_factor = params.max_norm / total_norm
        current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update

        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr

        if len(all_costs) == 100:
            logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                stidx, round(np.mean(all_costs), 2),
                int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                int(words_count * 1.0 / (time.time() - last_time)),
                round(100. * correct.item() / (stidx + k), 2)))
            print(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []
    train_acc = round(100 * correct.item() / len(s1), 2)
    print('results : epoch {0} ; mean accuracy train : {1}'.format(epoch, train_acc))
    return train_acc


def evaluate(epoch, eval_type='valid', final_eval=False):
    Insight_model.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    s1 = valid['s1'] if eval_type == 'valid' else test['s1']
    s2 = valid['s2'] if eval_type == 'valid' else test['s2']
    g1 = valid['g1'] if eval_type == 'valid' else test['g1']
    g2 = valid['g2'] if eval_type == 'valid' else test['g2']
    target = valid['label'] if eval_type == 'valid' else test['label']

    g1 = graphData(g1)
    g2 = graphData(g2)
    slices = g1.generate_batch(params.batch_size)

    for slice, i in zip(slices, range(0, len(s1), params.batch_size)):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()

        # model forward
        output = Insight_model((s1_batch, s1_len), (s2_batch, s2_len), g1, g2, slice, tgt_batch)

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

    # save model
    eval_acc = round(100 * correct.item() / len(s1), 2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :\
                  {2}'.format(epoch, eval_type, eval_acc))

    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(Insight_model.state_dict(), os.path.join(params.outputdir,
                                                          params.outputmodelname))
            val_acc_best = eval_acc
        else:
            if 'sgd' in params.optimizer:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
                print('Shrinking lr by : {0}. New lr = {1}'
                      .format(params.lrshrink,
                              optimizer.param_groups[0]['lr']))
                if optimizer.param_groups[0]['lr'] < params.minlr:
                    stop_training = True
            if 'adam' in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                stop_training = adam_stop
                adam_stop = True
    return eval_acc

"""
Train model
"""
epoch = 1

while not stop_training and epoch <= params.n_epochs:
    train_acc = trainepoch(epoch)
    eval_acc = evaluate(epoch, 'test')
    epoch += 1

# Run best model on test set
Insight_model.load_state_dict(torch.load(os.path.join(params.outputdir, params.outputmodelname)))

print('\nTEST : Epoch {0}'.format(epoch))
