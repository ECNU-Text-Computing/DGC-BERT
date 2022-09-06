import argparse
import datetime
import math
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from torch.nn import init

from pretrained_models.BAG import *
from torch import nn
import torch.nn.functional as F
from transformers import BertModel
from GNN.CustomConv import CustomGraphConv

from pretrained_models.dgc_bert import *

REL_POS = True


class BAGIGA(BAGIG):
    def __init__(self, vocab_size, embed_dim, num_class, pad_index, word2vec=None, keep_prob=0.5, pad_size=150,
                 hidden_size=768, model_path=None, mode='normal', model_type='BERT', ablation_module=None, **kwargs):
        super(BAGIGA, self).__init__(vocab_size, embed_dim, num_class, pad_index, word2vec, keep_prob, pad_size,
                                     hidden_size, model_path, mode, model_type, **kwargs)
        print('==current parent==', self.model_name)

        # print(self.bert_trans)
        self.model_name = self.model_name + '_ablation_' + ablation_module
        print('ablation test ' + ablation_module)
        if ablation_module == 'interaction':
            self.word_interaction = AblationInteractionModule(self.predict_dim, self.attention_mode)
            self.semantic_interaction = AblationInteractionModule(self.predict_dim, self.attention_mode)
        elif ablation_module == 'gate':
            self.word_gate = AblationGateModule(self.predict_dim)
            self.semantic_gate = AblationGateModule(self.predict_dim)
        elif ablation_module == 'appnp':
            self.gnn = AblationTopAttentionGNN(self.num_head, self.hidden_size, self.predict_dim, keep_prob,
                                               self.device,
                                               self.top_rate, 'GCN', self.reduce_method)
        elif ablation_module == 'mgcn':
            self.gnn = AblationTopAttentionGNN(self.num_head, self.hidden_size, self.predict_dim, keep_prob,
                                               self.device,
                                               self.top_rate, 'MGCN', self.reduce_method, self.k)
        elif ablation_module == 'gnn':
            self.gnn = AblationTopAttentionGNN(self.num_head, self.hidden_size, self.predict_dim, keep_prob,
                                               self.device,
                                               self.top_rate, 'simple', self.reduce_method)


class BAGIGS(BAGIGA):
    '''
    single layer prediction
    '''
    def __init__(self, vocab_size, embed_dim, num_class, pad_index, word2vec=None, keep_prob=0.5, pad_size=150,
                 hidden_size=768, model_path=None, mode='normal', model_type='BERT', ablation_module=None, **kwargs):
        super(BAGIGS, self).__init__(vocab_size, embed_dim, num_class, pad_index, word2vec, keep_prob, pad_size,
                                     hidden_size, model_path, mode, model_type, ablation_module, **kwargs)
        print('==current parent==', self.model_name)
        self.model_name = self.model_name + '_ablation_' + ablation_module
        self.final_dim = self.predict_dim if self.block_pooled else 2 * self.predict_dim
        self.gnn = SingleTopAttentionGNN(self.num_head, self.hidden_size, self.predict_dim, keep_prob, self.device,
                                         self.top_rate, ablation_module, self.reduce_method, self.k, self.alpha)
        self.single_gate = GateModule(self.predict_dim)
        print('ablation module', ablation_module)
        self.fc = nn.Sequential(
            FNN(self.final_dim, keep_prob, ACTIVATION),
            # nn.ReLU(),
            ACTIVATION(),
            nn.Dropout(p=keep_prob),
            nn.Linear(int(self.final_dim / 4), self.num_class),
        )

    def forward(self, content, lengths, masks, **kwargs):
        lengths = torch.sum(masks, dim=-1)
        content = content.permute(1, 0)

        output = self.bert(content, attention_mask=masks, return_dict=True, output_attentions=True,
                           output_hidden_states=True)
        pooled = output['pooler_output']

        single_attention_gnn = self.gnn(output, lengths)
        # print(single_attention_gnn.shape)
        if not self.block_pooled:
            bert_out = self.bert_trans(pooled)
            # word_attention_gnn_mix = self.word_interaction(word_attention_gnn, semantic_attention_gnn, masks)
            # semantic_attention_gnn_mix = self.semantic_interaction(semantic_attention_gnn, word_attention_gnn,  masks)

            single_attention_gnn = self.single_gate(bert_out, single_attention_gnn.mean(dim=1))

        # gnn_out = torch.cat((word_attention_gnn, semantic_attention_gnn), dim=1)
        gnn_out = single_attention_gnn
        if self.block_pooled:
            out = gnn_out
        else:
            out = torch.cat((bert_out, gnn_out), dim=1)
        # print(out.shape)

        out = self.dropout(out)
        out = self.fc(out)

        return out


class AblationInteractionModule(nn.Module):
    def __init__(self, dim_model, attention_mode='normal'):
        '''
        combine the information of word and semantic
        '''
        super(AblationInteractionModule, self).__init__()
        print('ablation interaction')
        self.dim_model = dim_model
        self.fc = nn.Linear(dim_model, dim_model)

    def forward(self, output, context, attention_mask):
        output = F.tanh(self.fc(output))
        output = output.mean(dim=1)
        return output


class AblationGateModule(nn.Module):
    def __init__(self, dim_model):
        '''
        let bert pooled filter the information
        '''
        super(AblationGateModule, self).__init__()
        self.gnn_trans = nn.Linear(dim_model, dim_model)
        self.activation = nn.Tanh()
        print('ablation gate')

    def forward(self, bert, gnn):
        out = self.activation(self.gnn_trans(gnn))
        return out


class AblationTopAttentionGNN(TopAttentionGNN):
    def __init__(self, num_head, dim_model, output_dim, keep_prob, device, top_rate=0.1, agg=None, reduce='mean', k=10):
        super(AblationTopAttentionGNN, self).__init__(num_head, dim_model, output_dim, keep_prob, device, top_rate, agg,
                                                      reduce, k)
        # self.word_gnn = TopAPPNPAttentionGNNModule(dim_model, keep_prob, self.device, top_rate, reduce)
        # self.semantic_gnn = TopAPPNPAttentionGNNModule(dim_model, keep_prob, self.device, top_rate, reduce)
        if agg == 'GCN':
            self.word_gnn = TopGCNAttentionGNNModule(dim_model, keep_prob, self.device, top_rate, reduce, output_dim)
            self.semantic_gnn = TopGCNAttentionGNNModule(dim_model, keep_prob, self.device, top_rate, reduce,
                                                         output_dim)
        elif agg == 'MGCN':
            self.word_gnn = TopGCNAttentionGNNModule(dim_model, keep_prob, self.device, top_rate, reduce, output_dim, k)
            self.semantic_gnn = TopGCNAttentionGNNModule(dim_model, keep_prob, self.device, top_rate, reduce,
                                                         output_dim, k)
        elif agg == 'simple':
            self.word_gnn = SimpleAttentionModule(dim_model, keep_prob, self.device, top_rate, reduce, output_dim)
            self.semantic_gnn = SimpleAttentionModule(dim_model, keep_prob, self.device, top_rate, reduce, output_dim)


class SingleTopAttentionGNN(TopAttentionGNN):
    def __init__(self, num_head, dim_model, output_dim, keep_prob, device, top_rate=0.1, agg=None, reduce='mean',
                 k=10, alpha=0.2):
        super(SingleTopAttentionGNN, self).__init__(num_head, dim_model, output_dim, keep_prob, device, top_rate, agg,
                                                    reduce)
        # self.word_gnn = TopAPPNPAttentionGNNModule(dim_model, keep_prob, self.device, top_rate, reduce)
        # self.semantic_gnn = TopAPPNPAttentionGNNModule(dim_model, keep_prob, self.device, top_rate, reduce)
        self.word_gnn = None
        self.semantic_gnn = None
        self.single_gnn = TopAPPNPAttentionGNNModule(dim_model, keep_prob, self.device, top_rate, reduce, output_dim,
                                                     k, alpha)
        self.agg = agg
        print('using single layer:', self.agg)

    def forward(self, output, lengths):
        if self.agg == 'word':
            word_attention = torch.stack(output['attentions'][:3], dim=4).max(dim=4)[0]
            word_embed = self.hs_word_trans(torch.stack(output['hidden_states'][:3], dim=3).transpose(-2, -1)
                                            ).transpose(-2, -1).max(dim=3)[0]
            output = self.activation(self.word_fc(self.single_gnn(word_embed, word_attention, lengths)))
        elif self.agg == 'semantic':
            semantic_attention = torch.stack(output['attentions'][-3:], dim=4).max(dim=4)[0]
            semantic_embed = self.hs_semantic_trans(torch.stack(output['hidden_states'][-4:-1], dim=3).transpose(-2, -1)
                                                    ).transpose(-2, -1).max(dim=3)[0]
            output = self.activation(self.semantic_fc(self.single_gnn(semantic_embed, semantic_attention, lengths)))

        return output


class TopGCNAttentionGNNModule(TopAPPNPAttentionGNNModule):
    def __init__(self, dim_model, keep_prob, device, top_rate=0.1, reduce='mean', predict_dim=None, k=1):
        super(TopGCNAttentionGNNModule, self).__init__(dim_model, keep_prob, device, top_rate, reduce, predict_dim)
        # print('--dim_model--', dim_model)
        # print('--predict_dim--', predict_dim)
        self.k = k
        if self.k == 1:
            self.conv = CustomGraphConv(predict_dim, predict_dim)
        else:
            convs = nn.ModuleList([CustomGraphConv(predict_dim, predict_dim) for i in range(k)])
            self.conv = convs
            print(self.conv)
        print('using gcn')

    def forward(self, hidden_state, attention, lengths):
        batch_size = hidden_state.size(0)
        seq_len = hidden_state.size(1)
        dealt_atten = attention.mean(dim=1)
        # dealt_atten = attention.max(dim=1)[0]
        topk_result = torch.topk(dealt_atten, round(self.top_rate * hidden_state.size(1)))
        topk_values = topk_result.values
        topk_indices = topk_result.indices
        sub_graphs = [self.seq_to_graph(topk_values[i], topk_indices[i], hidden_state[i], self.reduce, lengths[i]) for i in
                      range(batch_size)]
        batch_graph = dgl.batch(sub_graphs)

        if self.k == 1:
            result_node_embedding = self.conv(batch_graph, batch_graph.ndata['h'])
        else:
            result_node_embedding = batch_graph.ndata['h']
            for i in range(self.k):
                result_node_embedding = self.conv[i](batch_graph, result_node_embedding)

        batch_graph.ndata['h'] = result_node_embedding
        unbatched_graph = dgl.unbatch(batch_graph)
        # result_node_embedding = torch.cat([graph.ndata['h'].unsqueeze(dim=0) for graph in unbatched_graph], dim=0)
        result_node_embedding = torch.cat([
            torch.cat([graph.ndata['h'],
                       torch.zeros([(seq_len - graph.ndata['h'].shape[0]), self.predicut_dim]).to(self.device)],
                      dim=0).unsqueeze(dim=0)
            for graph in unbatched_graph], dim=0)
        out = self.fc(self.dropout(self.activation(result_node_embedding)))
        out = self.ln(out)
        return out


class SimpleAttentionModule(nn.Module):
    def __init__(self, dim_model, keep_prob, device, top_rate=0.1, reduce='mean', predict_dim=None):
        super(SimpleAttentionModule, self).__init__()
        self.hs_fc = nn.Linear(dim_model, predict_dim)
        self.dropout = nn.Dropout(keep_prob)
        self.ln = nn.LayerNorm(predict_dim)
        self.fc = nn.Linear(predict_dim, predict_dim)
        self.activation = nn.Tanh()
        print('using simple fnn')

    def forward(self, hidden_state, attention, lengths):
        result_node_embedding = torch.bmm(attention.mean(dim=1), self.hs_fc(hidden_state))
        out = self.fc(self.dropout(self.activation(result_node_embedding)))
        out = self.ln(out)
        return out


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

    args = {'k': 5, 'alpha': None, 'top_rate': 0.1, 'predict_dim': None}
    model = BAGIGA(50000, 768, 2, 0, 512, model_path='../bert/base_bert/', mode='top_biaffine+softmax',
                   ablation_module='mgcn', args=args).cuda()
    # model = BAGIGS(50000, 768, 2, 0, 512, model_path='../bert/base_bert/', mode='top_biaffine+softmax',
    #                ablation_module='semantic', args=args).cuda()
    seq_len = 128
    bs = 2
    # x = torch.randint(0, 20000, (seq_len, 2)).cuda()
    # x = torch.ones((seq_len, 2)).cuda()
    # x = torch.zeros(256, 2, dtype=torch.int).cuda()
    x = torch.tensor(([list(range(seq_len * i, seq_len * (i + 1) - i * 10)) + [0] * 10 * i for i in range(bs)]),
                     dtype=torch.int).permute(1,
                                              0).cuda()
    lengths = torch.tensor([seq_len * (i + 1) - i * 10 - seq_len * i for i in range(bs)]).cuda()
    masks = torch.tensor([[1] * length + [0] * (seq_len - length) for length in lengths]).cuda()
    print(lengths)
    print(x.shape)
    print(model(x, lengths, masks).cuda())
