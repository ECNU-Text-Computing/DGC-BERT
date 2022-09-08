import json
import math

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from sklearn.metrics import precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_auc_score
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer, BertTokenizer
from scipy import stats

# word_result = torch.tensor(
#     [[1, 2, 3, 4, 5],
#      [5, 5, 4, 3, 5]], dtype=torch.float32
# ).unsqueeze(dim=0)
# semantic_result = torch.tensor(
#     [[1, 7, 3, 5, 5],
#      # [1, 7, 8, 3, 5],
#      [5, 5, 6, 7, 0]], dtype=torch.float32
# ).unsqueeze(dim=0)
#
# bilinear = nn.Bilinear(5, 5, 2)
# print(word_result.shape)
# print(semantic_result.shape)
# print(bilinear(word_result, semantic_result).shape)
from torch.nn import init

# result = torch.load('mistake_results_BAGIG.list')
# index = 15
# print(result[index])
# print(len(result))
# probit, label, true_label, indexes, state = zip(*result)
# print(indexes[index])
#
# with open('./data/PeerRead/content.list') as fr:
#     contents = fr.readlines()
# with open('./data/PeerRead/label.list') as fr:
#     labels = fr.readlines()
# print(contents[result[index][3]].strip())
# print(labels[result[index][3]].strip())


# tokenizer = BertTokenizer.from_pretrained('./bert/scibert/vocab.txt')
# # tokenizer.convert_ids_to_tokens()
# # bert = BertModel.from_pretrained('./bert/scibert/', return_dict=True, output_attentions=True, output_hidden_states=True).cuda()
# sample_string = 'maximum likelihood and maximum a posteriori direction-of-arrival estimation in the presence of ' \
#                 'sirp noise. the maximum likelihood ( ml ) and maximum a posteriori ( map ) estimation techniques ' \
#                 'are widely used to address the direction of arrival ( doa ) estimation problems an important ' \
#                 'topic in sensor array processing. conventionally the ml estimators in the doa estimation context ' \
#                 'assume the sensor noise to follow a gaussian distribution. in real life application however this ' \
#                 'assumption is sometimes not valid and it is often more accurate to model the noise as a non ' \
#                 'gaussian process. in this paper we derive an iterative ml as well as an iterative map estimation ' \
#                 'algorithm for the doa estimation problem under the spherically invariant random process noise ' \
#                 'assumption one of the most popular non gaussian models especially in the radar context. ' \
#                 'numerical simulation results are provided to assess our proposed algorithms and to show their ' \
#                 'maximum likelihood and maximum a posteriori direction-of-arrival estimation in the presence of ' \
#                 'sirp noise. the maximum likelihood ( ml ) and maximum a posteriori ( map ) estimation techniques ' \
#                 'are widely used to address the direction of arrival ( doa ) estimation problems an important ' \
#                 'topic in sensor array processing. conventionally the ml estimators in the doa estimation context ' \
#                 'assume the sensor noise to follow a gaussian distribution. in real life application however this ' \
#                 'assumption is sometimes not valid and it is often more accurate to model the noise as a non ' \
#                 'gaussian process. in this paper we derive an iterative ml as well as an iterative map estimation ' \
#                 'algorithm for the doa estimation problem under the spherically invariant random process noise ' \
#                 'assumption one of the most popular non gaussian models especially in the radar context. ' \
#                 'numerical simulation results are provided to assess our proposed algorithms and to show their ' \
#                 'maximum likelihood and maximum a posteriori direction-of-arrival estimation in the presence of ' \
#                 'sirp noise. the maximum likelihood ( ml ) and maximum a posteriori ( map ) estimation techniques ' \
#                 'are widely used to address the direction of arrival ( doa ) estimation problems an important ' \
#                 'topic in sensor array processing. conventionally the ml estimators in the doa estimation context ' \
#                 'assume the sensor noise to follow a gaussian distribution. in real life application however this ' \
#                 'assumption is sometimes not valid and it is often more accurate to model the noise as a non ' \
#                 'gaussian process. in this paper we derive an iterative ml as well as an iterative map estimation ' \
#                 'algorithm for the doa estimation problem under the spherically invariant random process noise ' \
#                 'assumption one of the most popular non gaussian models especially in the radar context. ' \
#                 'numerical simulation results are provided to assess our proposed algorithms and to show their ' \
#                 'maximum likelihood and maximum a posteriori direction-of-arrival estimation in the presence of ' \
#                 'sirp noise. the maximum likelihood ( ml ) and maximum a posteriori ( map ) estimation techniques ' \
#                 'are widely used to address the direction of arrival ( doa ) estimation problems an important ' \
#                 'topic in sensor array processing. conventionally the ml estimators in the doa estimation context ' \
#                 'assume the sensor noise to follow a gaussian distribution. in real life application however this ' \
#                 'assumption is sometimes not valid and it is often more accurate to model the noise as a non ' \
#                 'gaussian process. in this paper we derive an iterative ml as well as an iterative map estimation ' \
#                 'algorithm for the doa estimation problem under the spherically invariant random process noise ' \
#                 'assumption one of the most popular non gaussian models especially in the radar context. ' \
#                 'numerical simulation results are provided to assess our proposed algorithms and to show their '
# # tokens = tokenizer(sample_string)['input_ids']
# # print(len(tokens))
# tokens = tokenizer.tokenize(sample_string)
# print(tokenizer(sample_string)['input_ids'])
# print(tokens)
# print(tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]']))
# print(np.arange(0, 1000, 460))
# print([1,2,3][1:5])
# masks = tokenizer(sample_string)['attention_mask']
# seq_len = 256
# threshold = 0.1
# masks.extend([0] * (seq_len - len(tokens)))
# tokens.extend([0] * (seq_len - len(tokens)))
# tokens = torch.tensor(tokens).unsqueeze(dim=0).cuda()
# masks = torch.tensor(masks).unsqueeze(dim=0).cuda()
# print(tokens.shape)
# print(masks.shape)
# print(masks.sum())
# output = bert(tokens, attention_mask=masks)
# # selected_attention = output['attentions'][0].mean(dim=1)
# print(output['attentions'][-1][0][0][masks.sum()-1][masks.sum():])
# print(output['attentions'][-1].sum(dim=-1))
# selected_attention = output['attentions'][0].max(dim=1)[0]
# # selected_attention = torch.stack(output['attentions'][:3], dim=4).mean(dim=4).max(dim=1)[0]
# print(selected_attention.shape)
# all_attention = torch.flatten(selected_attention)
# df = pd.DataFrame(all_attention.detach().cpu().numpy()).reset_index()
# print(len(df))
# df = df[df[0] > 0]
# print(len(df))
# df = df[df[0] >= threshold]
# print(len(df))
# df[0].hist(bins=100)
# plt.savefig('attention_dist.png')
#
# print(torch.sum(selected_attention > threshold, dim=-1))
# print(torch.sum(selected_attention > threshold))

# print(selected_attention[0][0].shape)
# print(torch.gather(selected_attention[0][0], 0, torch.tensor([5, 6, 7]).cuda()))

# x = torch.randn(32, 128, 256).cuda()
# index = torch.tensor([10,20,30]).long()
# print(x[[10,20,30]].shape)
#
# i = torch.LongTensor([[0, 1, 1],
#                       [2, 0, 2]])
# v = torch.FloatTensor([3, 4, 5])
# print(torch.sparse.FloatTensor(i, v, torch.Size([2,3])).to_dense())

# x = torch.randn(32, 256)
# y = torch.randn(32, 256)
# print(torch.stack([x, y], dim=1).shape)


# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# print(tokenizer.vocab)

# index = torch.tensor([
#     [1, 2, 5, 7],
#     [2, 3, 4, 5],
#     [4, 1, 2, 5],
# ])
#
# value = torch.tensor([
#     [0.1, 0.2, 0.5, 0],
#     [0.5, 0.7, 0, 0.1],
#     [0.4, 0.1, 0.2, 0],
# ], dtype=torch.float)
#
# top_num = index.shape[1]
# node_num = index.shape[0]
#
# target_nodes = torch.tensor([[i] * top_num for i in range(node_num)])
# print(target_nodes)
# mask_value = F.softmax(torch.where(value > 0.0, value, torch.tensor(-1e9, dtype=torch.float)), dim=-1)
# print(mask_value)
#
# print(torch.flatten(mask_value))
# pos_sign = torch.nonzero(torch.flatten(mask_value), as_tuple=True)[0]
# print(pos_sign)
#
# edge_tuple = (torch.flatten(index)[pos_sign], torch.flatten(target_nodes)[pos_sign])
#
# print(edge_tuple[0])
# graph = dgl.graph(edge_tuple)
# print(graph)
#
# print(coo_matrix(
#                 ([0], (np.array([0]), np.array([0], dtype=int))),
#                 shape=(1, 5000), dtype=float
#             ).todense())
# x = np.array([[0.2] * 5])
# y = np.array([[0.25] * 4 + [0]])
# print(x)
# print(stats.entropy(y[0], x[0]))
# x = F.softmax(torch.from_numpy(x), dim=-1)
# y = F.softmax(torch.from_numpy(y), dim=-1)
# print(stats.entropy(y[0].numpy(), x[0].numpy()))
# print(F.kl_div(x.log(), y, reduction='sum'))

# x = torch.tensor([
#     [1, 1, 1],
#     [2, 0, 3],
#     [0, 5, 2],
#     [3, 3, 3]
# ], dtype=torch.float).unsqueeze(0).unsqueeze(0)
# print(x.shape)
# pool = nn.MaxPool2d((4, 3))
# print(pool(x))
# print(pool(x).shape)
# print(x.max(dim=2)[0])

# x = torch.tensor([
#     [1, 1, 1],
#     [2, 0, 3],
#     [0, 5, 2],
#     [3, 3, 3]
# ], dtype=torch.float)
# print(x.shape)
# print(x[[0, 1], [1,2]])
# x[[0, 1], [1,2]] = torch.tensor([2, 4]).float()
# print(x)
from main import get_configs, get_DL_data, get_model

data_source = 'AAPR'
seed = 555
model_config = get_configs(data_source, ['DGCBERT'])['DGCBERT']
pretrained_types = json.load(open('./configs/pretrained_types.json', 'r'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_config['model_path'] = pretrained_types['SciBERT']['model_path']
model_config['vocab_path'] = pretrained_types['SciBERT']['vocab_path']
model_config['mode'] = 'top_biaffine+softmax'
print(model_config)
dataProcessor = get_DL_data(base_config=model_config, data_source=data_source,
                            BERT_tokenizer_path=model_config['vocab_path'],
                            load_data=model_config['saved_data'], seed=seed)
# print(dataProcessor.get_dealt_text('you know this is test', 16, True)[0])

# args = {'k': 5, 'alpha': 0.5, 'top_rate': 0.3, 'predict_dim': 128}
# args = {'k': 10, 'alpha': 0.2, 'top_rate': 0.05, 'predict_dim': 256}
# model = get_model('DGCBERT', dataProcessor, device, model_config, args=args)
model = get_model('SciBERT', dataProcessor, device, model_config)
# model.load_state_dict(torch.load('./checkpoints/AAPR/BAGIG_top_biaffine+softmax_SciBERT_3_666.pkl'))
# model = torch.load('./checkpoints/PeerRead/BAGIG_top_biaffine+softmax_SciBERT_4_better_codes.pkl').cuda()
# model = torch.load('./checkpoints/AAPR/BAGIG_top_biaffine+softmax_SciBERT_3_666.pkl').cuda()
# model = torch.load('./checkpoints/AAPR/SciBERT_2.pkl').cuda()
# model = torch.load('./checkpoints/PeerRead/SciBERT_4.pkl').cuda()
model.mode = None


# print(state_dict['fc.2.weight'])
# state_dict = model.state_dict()
# state_dict['fc.2.weight'] = state_dict['fc.3.weight']
# state_dict['fc.2.bias'] = state_dict['fc.3.bias']
# state_dict.pop('fc.3.weight')
# state_dict.pop('fc.3.bias')

# torch.save({'state_dict': state_dict}, './checkpoints/{}/DGC_BERT_best.pt'.format(data_source))
# torch.save({'state_dict': state_dict}, './checkpoints/{}/SciBERT_best.pt'.format(data_source))

# state_dict = torch.load('./checkpoints/{}/DGC_BERT_best.pt'.format(data_source))['state_dict']
state_dict = torch.load('./checkpoints/{}/SciBERT_best.pt'.format(data_source))['state_dict']
# print(state_dict.keys())
model.load_state_dict(state_dict)

model.test(dataProcessor.dataloaders[2])
# print(model.state_dict().keys())
# torch.save({'state_dict': model.state_dict()}, './checkpoints/AAPR/state_dict')
