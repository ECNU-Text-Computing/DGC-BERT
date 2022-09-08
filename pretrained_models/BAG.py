import argparse
import datetime
import os
import random

import numpy as np
import torch
import dgl
# from dgl.nn.pytorch import APPNPConv, TAGConv
from GNN.CustomConv import CustomAPPNPConv, CustomTAGConv
from dgl.nn.pytorch.conv import SAGEConv

from torch import nn
import torch.nn.functional as F
from pretrained_models.base_bert import BaseBert
# from pretrained_models.bert_attention_cnn import FNN

# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOP_RATE = 0.05
FILTER_THRESHOLD = 0.1
ACTIVATION = nn.Tanh
print(ACTIVATION)


# AGG = 'APPNP'
# print('agg_method', AGG)

class FNN(nn.Module):
    def __init__(self, input_dim, keep_prob, activation=nn.Tanh):
        '''
        :param input_dim:
        '''
        super(FNN, self).__init__()
        self.input_dim = input_dim
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, int(self.input_dim / 2)),
            # nn.ReLU(),
            activation(),
            # nn.Dropout(p=keep_prob),
            nn.Linear(int(self.input_dim / 2), int(self.input_dim / 4))
        )
        self.output_dim = int(self.input_dim / 4)

    def forward(self, x):
        out = self.fc(x)
        return out


class BAG(BaseBert):
    def __init__(self, vocab_size, embed_dim, num_class, pad_index, word2vec=None, keep_prob=0.5, pad_size=150,
                 hidden_size=768, model_path=None, mode=None, model_type='BERT', **kwargs):
        super(BAG, self).__init__(vocab_size, embed_dim, num_class, pad_index, pad_size, word2vec,
                                  keep_prob, hidden_size, model_path, **kwargs)
        self.model_name = 'BAG'
        self.predict_dim = 64
        # print('torch.backends.cudnn.benchmark:', torch.backends.cudnn.benchmark)
        # print('torch.backends.cudnn.deterministic:', torch.backends.cudnn.deterministic)
        print('predict_dim', self.predict_dim)
        self.block_pooled = False
        self.reduce_method = kwargs['reduced_method'] if 'reduced_method' in kwargs.keys() else 'mean'
        print('block_pooled', self.block_pooled)
        self.top_rate = TOP_RATE
        if mode:
            self.model_name = self.model_name + '_' + mode + '_' + model_type
            print('using ' + mode)
        else:
            self.model_name = self.model_name + '_' + model_type
        if mode == 'top':
            self.gnn = TopGNN(self.num_head, self.hidden_size, self.predict_dim, keep_prob, self.device, self.top_rate,
                              'base', self.reduce_method)
        elif mode == 'top_sage':
            self.gnn = TopGNN(self.num_head, self.hidden_size, self.predict_dim, keep_prob, self.device, self.top_rate,
                              'SAGE', self.reduce_method)
        elif mode == 'top_appnp':
            self.gnn = TopGNN(self.num_head, self.hidden_size, self.predict_dim, keep_prob, self.device, self.top_rate,
                              'APPNP', self.reduce_method)
        elif mode == 'top_appnp+softmax':
            self.gnn = TopGNN(self.num_head, self.hidden_size, self.predict_dim, keep_prob, self.device, self.top_rate,
                              'APPNP', 'softmax')
        elif mode == 'top_tag':
            self.gnn = TopGNN(self.num_head, self.hidden_size, self.predict_dim, keep_prob, self.device, self.top_rate,
                              'TAG', self.reduce_method)
        else:
            self.gnn = BaseGNN(self.num_head, self.hidden_size, self.predict_dim, keep_prob)

        # self.final_dim = self.hidden_size + self.cnn.out_dim
        self.final_dim = 2 * self.predict_dim if self.block_pooled else 3 * self.predict_dim
        self.bert_trans = nn.Sequential(
            # nn.Dropout(p=keep_prob),
            nn.Linear(self.hidden_size, self.predict_dim),
            # nn.ReLU(),
            ACTIVATION()
        )
        # print(self.bert_trans)
        self.fc = nn.Sequential(
            FNN(self.final_dim, keep_prob, ACTIVATION),
            # nn.ReLU(),
            ACTIVATION(),
            # nn.Dropout(p=keep_prob),
            nn.Linear(int(self.final_dim / 4), self.num_class),
        )
        # print(self.fc)
        # self.fc = nn.Linear(self.final_dim, self.num_class)
        self.adaptive_lr = True

    def forward(self, content, lengths, masks, **kwargs):
        lengths = torch.sum(masks, dim=-1)
        content = content.permute(1, 0)

        output = self.bert(content, attention_mask=masks, return_dict=True, output_attentions=True,
                           output_hidden_states=True)
        pooled = output['pooler_output']

        word_attention_gnn, semantic_attention_gnn = self.gnn(output, lengths)
        if not self.block_pooled:
            bert_out = self.bert_trans(pooled)

        gnn_out = torch.cat((word_attention_gnn, semantic_attention_gnn), dim=1)
        if self.block_pooled:
            out = gnn_out
        else:
            out = torch.cat((bert_out, gnn_out), dim=1)

        # out = word_attention_cnn + semantic_attention_cnn + bert_out

        out = self.dropout(out)
        out = self.fc(out)
        # out = F.softmax(out, dim=1)

        return out


class BaseGNN(nn.Module):
    def __init__(self, num_head, dim_model, output_dim, keep_prob):
        super(BaseGNN, self).__init__()
        self.word_gnn = BaseGNNModule(num_head, dim_model, keep_prob)
        self.semantic_gnn = BaseGNNModule(num_head, dim_model, keep_prob)
        self.activation = ACTIVATION()
        self.word_fc = nn.Linear(dim_model, output_dim)
        self.semantic_fc = nn.Linear(dim_model, output_dim)
        # print('change single linear to multi linear')
        # self.word_fc = nn.Sequential(
        #     FNN(dim_model, keep_prob),
        #     ACTIVATION(),
        #     nn.Linear(int(dim_model/4), output_dim),
        # )
        # self.semantic_fc = nn.Sequential(
        #     FNN(dim_model, keep_prob),
        #     ACTIVATION(),
        #     nn.Linear(int(dim_model/4), output_dim),
        # )

    def forward(self, output, lengths):
        word_attention = output['attentions'][0]
        word_embed = output['hidden_states'][1]
        semantic_attention = output['attentions'][-1]
        semantic_embed = output['hidden_states'][-1]
        word_output = self.activation(self.word_fc(self.word_gnn(word_embed, word_attention)))
        semantic_output = self.activation(self.semantic_fc(self.semantic_gnn(semantic_embed, semantic_attention)))

        return word_output, semantic_output


class BaseGNNModule(nn.Module):
    def __init__(self, num_head, dim_model, keep_prob):
        super(BaseGNNModule, self).__init__()
        self.num_head = num_head
        self.dim_head = dim_model // num_head
        self.trans = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(keep_prob)
        self.activation = ACTIVATION()
        self.ln = nn.LayerNorm(dim_model)

    def forward(self, hidden_state, attention):
        batch_size = hidden_state.size(0)
        seq_len = attention.size(2)
        hidden_state = self.trans(hidden_state).view(batch_size * self.num_head, -1, self.dim_head)
        attention = attention.view(batch_size * self.num_head, seq_len, seq_len)
        hidden_state = self.activation(
            torch.matmul(attention, hidden_state).view(batch_size, -1, self.dim_head * self.num_head))
        hidden_state = torch.mean(self.fc(self.dropout(hidden_state)), dim=1)
        hidden_state = self.ln(hidden_state)
        return hidden_state


class TopGNN(BaseGNN):
    def __init__(self, num_head, dim_model, output_dim, keep_prob, device, top_rate=0.1, agg=None, reduce='mean'):
        super(TopGNN, self).__init__(num_head, dim_model, output_dim, keep_prob)
        print('top rate', top_rate)
        print('reduce_method', reduce)
        self.word_fc = nn.Linear(output_dim, output_dim)
        self.semantic_fc = nn.Linear(output_dim, output_dim)
        self.device = device
        if agg == 'SAGE':
            self.GNNModule = TopSAGEGNNModule
        elif agg == 'APPNP':
            self.GNNModule = TopAPPNPGNNModule
        elif agg == 'TAG':
            self.GNNModule = TopTAGGNNModule
        else:
            self.GNNModule = TopGNNModule
        self.word_gnn = self.GNNModule(dim_model, keep_prob, self.device, top_rate, reduce, output_dim)
        self.semantic_gnn = self.GNNModule(dim_model, keep_prob, self.device, top_rate, reduce, output_dim)

    def forward(self, output, lengths):
        print(lengths)
        word_attention = output['attentions'][0]
        word_embed = output['hidden_states'][0]
        semantic_attention = output['attentions'][-1]
        semantic_embed = output['hidden_states'][-2]
        word_output = self.activation(self.word_fc(self.word_gnn(word_embed, word_attention, lengths)))
        semantic_output = self.activation(self.semantic_fc(self.semantic_gnn(semantic_embed, semantic_attention, lengths)))

        return word_output, semantic_output


def my_message_function(edges):
    return {'weighted_message': edges.src['h'] * edges.data['w'].unsqueeze(1).expand_as(edges.src['h'])}


def my_reduce_fcuntion(nodes):
    return {'h': torch.mean(nodes.mailbox['weighted_message'], dim=1)}


class TopGNNModule(nn.Module):
    def __init__(self, dim_model, keep_prob, device, top_rate=0.1, reduce='mean', predict_dim=None):
        super(TopGNNModule, self).__init__()
        self.device = device
        self.dim_model = dim_model
        self.top_rate = top_rate
        self.predict_dim = predict_dim
        # self.node_trans = nn.Linear(self.dim_model, self.dim_model)
        self.node_trans = nn.Linear(self.dim_model, self.predict_dim)
        # self.node_trans1 = nn.Linear(self.dim_model, self.dim_model)
        # self.node_trans2 = nn.Linear(self.dim_model, self.dim_model)
        # self.fc = nn.Linear(self.dim_model, self.dim_model)
        self.fc = nn.Linear(self.predict_dim, self.predict_dim)
        self.dropout = nn.Dropout(keep_prob)
        self.activation = ACTIVATION()
        # self.ln = nn.LayerNorm(self.dim_model)
        self.ln = nn.LayerNorm(self.predict_dim)
        self.node_eta = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda()
        self.node_eta.data.fill_(0)
        self.reduce = reduce
        # self.relative_position_embed = RelativePosition(self.dim_model, 32)

    def seq_to_graph(self, topk_value, topk_indice, nodes, reduce='mean', length=256):
        '''
        :param topk_value:
        :param topk_indice:
        :param reduce:
        :param nodes:
        :return:
        '''

        valid_topk_value = topk_value[:length]
        valid_topk_indice = topk_indice[:length]
        top_num = valid_topk_indice.shape[1]
        target_nodes = torch.tensor([[i] * top_num for i in range(length)]).to(self.device)
        # attention mask
        mask_edge_value = F.softmax(torch.where(valid_topk_value > 0.0, valid_topk_value,
                                           torch.tensor(-1e9, dtype=torch.float).to(self.device)), dim=-1)
        pos_sign = torch.nonzero(torch.flatten(mask_edge_value), as_tuple=True)[0]
        source_nodes = torch.flatten(valid_topk_indice)[pos_sign]
        target_nodes = torch.flatten(target_nodes)[pos_sign]
        edge_tuple = (source_nodes, target_nodes)
        if reduce == 'softmax':
            edge_weight = torch.flatten(mask_edge_value)[pos_sign]

        sub_graph = dgl.graph(edge_tuple).to(self.device)
        # sub_graph.ndata['h'] = nodes
        sub_graph.ndata['h'] = self.node_trans(nodes[:length])
        # sub_graph.ndata['h'] = self.node_trans2(self.dropout(self.activation(self.node_trans1(nodes))))
        # sub_graph.ndata['index'] = torch.tensor(range(nodes.shape[0])).to(self.device)
        sub_graph.ndata['index'] = torch.tensor(range(length)).to(self.device)
        if reduce == 'mean':
            sub_graph.edata['w'] = torch.tensor([1] * len(edge_tuple[0]), dtype=torch.float32).to(self.device)
        elif reduce == 'softmax':
            # edge_weight = torch.cat(edge_weight)
            sub_graph.edata['w'] = edge_weight
        return sub_graph

    def forward(self, hidden_state, attention, lengths):
        print(lengths)
        batch_size = hidden_state.size(0)
        avg_atten = attention.mean(dim=1)
        topk_results = torch.topk(avg_atten, round(self.top_rate * hidden_state.size(1)))
        # relative_position = self.relative_position_embed(hidden_state.size(1), hidden_state.size(1))
        # print('top_k:\n', topk_result)
        topk_values = topk_results.values
        topk_indices = topk_results.indices
        sub_graphs = [self.seq_to_graph(topk_values[i], topk_indices[i], hidden_state[i], self.reduce, lengths[i])
                      for i in range(batch_size)]
        batch_graph = dgl.batch(sub_graphs)
        before_node_embedding = batch_graph.ndata['h']
        batch_graph.update_all(
            message_func=my_message_function,
            reduce_func=my_reduce_fcuntion
        )

        # batch_graph.ndata['h'] = batch_graph.ndata['h'].float()
        # batch_graph.edata['w'] = batch_graph.edata['w'].float()
        after_node_embedding = batch_graph.ndata['h']

        new_node_embedding = self.node_eta * before_node_embedding + (1 - self.node_eta) * after_node_embedding
        batch_graph.ndata['h'] = new_node_embedding

        out = dgl.mean_nodes(batch_graph, feat='h')
        # print('out:\n', out)
        out = self.fc(self.dropout(self.activation(out)))
        out = self.ln(out)
        return out


class TopSAGEGNNModule(TopGNNModule):
    def __init__(self, dim_model, keep_prob, device, top_rate=0.1, reduce='mean', predict_dim=None):
        super(TopSAGEGNNModule, self).__init__(dim_model, keep_prob, device, top_rate, reduce, predict_dim)
        self.conv = SAGEConv(predict_dim, predict_dim, 'mean')

    def forward(self, hidden_state, attention, lengths):
        batch_size = hidden_state.size(0)
        dealt_atten = attention.mean(dim=1)
        # dealt_atten = attention.max(dim=1)[0]
        topk_result = torch.topk(dealt_atten, round(self.top_rate * hidden_state.size(1)))
        topk_values = topk_result.values
        topk_indices = topk_result.indices
        # self.conv.rel_pos_embed = self.relative_position_embed(hidden_state.size(1), hidden_state.size(1))
        sub_graphs = [self.seq_to_graph(topk_values[i], topk_indices[i], hidden_state[i], self.reduce, lengths[i]) for i in
                      range(batch_size)]
        batch_graph = dgl.batch(sub_graphs)

        result_node_embedding = self.conv(batch_graph, batch_graph.ndata['h'])

        batch_graph.ndata['h'] = result_node_embedding
        # out = dgl.mean_nodes(batch_graph, feat='h')
        out = dgl.sum_nodes(batch_graph, feat='h')
        out = (out.T / lengths).T
        out = self.fc(self.dropout(self.activation(out)))
        out = self.ln(out)
        return out


class TopAPPNPGNNModule(TopSAGEGNNModule):
    def __init__(self, dim_model, keep_prob, device, top_rate=0.1, reduce='mean', predict_dim=None, k=10, alpha=0.2):
        super(TopAPPNPGNNModule, self).__init__(dim_model, keep_prob, device, top_rate, reduce, predict_dim)
        self.conv = CustomAPPNPConv(k=k, alpha=alpha)
        # print(self.conv._k)
        # print(self.conv._alpha)
        # print(self.conv.edge_drop)


class TopTAGGNNModule(TopSAGEGNNModule):
    def __init__(self, dim_model, keep_prob, device, top_rate=0.1, reduce='mean', predict_dim=None):
        super(TopTAGGNNModule, self).__init__(dim_model, keep_prob, device, top_rate, reduce, predict_dim)
        self.conv = CustomTAGConv(predict_dim, predict_dim, k=3)


if __name__ == "__main__":
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        dgl.seed(seed)
        # torch.backends.cudnn.deterministic = True
        os.environ["OMP_NUM_THREADS"] = '1'


    setup_seed(123)
    # sample = {}
    # sample['attentions'] = [torch.randn(2, 12, 256, 256).cuda()] * 2
    # sample['hidden_states'] = [torch.randn(2, 256, 768).cuda()] * 3
    # model = BaseGNN(12, 768, 64, 0.5).cuda()
    # result = model(sample)
    # print(result[0].shape)
    # print(result[0])
    model = BAG(50000, 768, 2, 0, 512, model_path='../bert/base_bert/', mode='top_appnp+softmax').cuda()
    seq_len = 128
    bs = 2
    # x = torch.randint(0, 20000, (seq_len, 2)).cuda()
    # x = torch.ones((seq_len, 2)).cuda()
    # x = torch.zeros(256, 2, dtype=torch.int).cuda()
    # x = torch.tensor(([list(range(seq_len * i, seq_len * (i + 1))) for i in range(bs)]), dtype=torch.int).permute(1,
    #                                                                                                               0).cuda()
    x = torch.tensor(([list(range(seq_len * i, seq_len * (i + 1) - i * 10)) + [0] * 10 * i for i in range(bs)]), dtype=torch.int).permute(1,
                                                                                                                  0).cuda()
    lengths = torch.tensor([seq_len * (i + 1) - i * 10 - seq_len * i for i in range(bs)]).cuda()
    masks = torch.tensor([[1] * length + [0] * (seq_len-length) for length in lengths]).cuda()
    # print(lengths)
    # print(x.shape)
    print(model(x, lengths, masks).cuda())
