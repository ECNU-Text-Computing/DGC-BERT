import argparse
import datetime
import math
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from torch.nn import init

from models.base_model import get_format_str, cal_metrics
from pretrained_models.BAG import *
from torch import nn
import torch.nn.functional as F
from transformers import BertModel

from pretrained_models.BAGT import BAGT
from pretrained_models.dgcbert import DGCBERT

REL_POS = False


# ACTIVATION = nn.Tanh


class DGCBERTO(DGCBERT):
    def __init__(self, vocab_size, embed_dim, num_class, pad_index, word2vec=None, keep_prob=0.5, pad_size=150,
                 hidden_size=768, model_path=None, mode='normal', model_type='BERT', **kwargs):
        super(DGCBERTO, self).__init__(vocab_size, embed_dim, num_class, pad_index, word2vec, keep_prob, pad_size,
                                      hidden_size, model_path, mode, model_type, **kwargs)
        print('==current parent==', self.model_name)
        self.model_name = 'DGCBERT'
        self.block_pooled = False
        self.reduce_method = 'mean'
        self.predict_dim = 128
        self.k = 10
        self.alpha = 0.2
        self.attention_mode = 'biaffine'
        if 'args' in kwargs:
            self.args = kwargs['args']
            print(self.args)
            self.k = int(self.args['k']) if self.args['k'] else self.k
            self.alpha = float(self.args['alpha']) if self.args['alpha'] else self.alpha
            self.top_rate = float(self.args['top_rate']) if self.args['top_rate'] else self.top_rate
            self.predict_dim = int(self.args['predict_dim']) if self.args['predict_dim'] else self.predict_dim
        print('predict_dim', self.predict_dim)
        if mode:
            self.model_name = self.model_name + '_' + mode + '_' + model_type
            print('using ' + mode)
        else:
            self.model_name = self.model_name + '_' + model_type
        # print('top rate', TOP_RATE)
        if mode == 'top_normal':
            self.attention_mode = 'normal'
            self.gnn = TopAttentionGNN(self.num_head, self.hidden_size, self.predict_dim, keep_prob, self.device,
                                       self.top_rate, 'APPNP', self.reduce_method, self.k, self.alpha)
        elif mode == 'top_biaffine':
            self.attention_mode = 'biaffine'
            self.gnn = TopAttentionGNN(self.num_head, self.hidden_size, self.predict_dim, keep_prob, self.device,
                                       self.top_rate, 'APPNP', self.reduce_method, self.k, self.alpha)
        elif mode == 'top_biaffine+softmax':
            self.attention_mode = 'biaffine'
            self.reduce_method = 'softmax'
            self.gnn = TopAttentionGNN(self.num_head, self.hidden_size, self.predict_dim, keep_prob, self.device,
                                       self.top_rate, 'APPNP', self.reduce_method, self.k, self.alpha)

        self.final_dim = 2 * self.predict_dim if self.block_pooled else 3 * self.predict_dim
        # if self.predict_dim <= 196:
        if self.predict_dim <= 512:
            print('compressed')
            self.bert_trans = nn.Sequential(
                # nn.Dropout(p=keep_prob),
                FNN(self.hidden_size, keep_prob, ACTIVATION),
                ACTIVATION(),
                nn.Dropout(p=keep_prob),
                nn.Linear(int(self.hidden_size / 4), self.predict_dim),
                ACTIVATION()
            )
        elif self.predict_dim == 768:
            print('uncompressed')
            self.bert_trans = nn.Sequential(
                nn.Dropout(p=keep_prob),
                nn.Linear(self.hidden_size, self.hidden_size),
                ACTIVATION()
            )
        else:
            print('some situations')
            self.bert_trans = nn.Sequential(
                # nn.Dropout(p=keep_prob),
                nn.Linear(self.hidden_size, int(self.hidden_size / 2)),
                # ACTIVATION(,
                ACTIVATION(),
                # nn.Dropout(p=keep_prob),
                nn.Linear(int(self.hidden_size / 2), int(self.hidden_size / 2)),
                # ACTIVATION(,
                ACTIVATION(),
                nn.Dropout(p=keep_prob),
                nn.Linear(int(self.hidden_size / 2), self.predict_dim),
                # ACTIVATION(
                ACTIVATION()
            )

    def forward(self, content, lengths, masks, **kwargs):
        lengths = torch.sum(masks, dim=-1)
        content = content.permute(1, 0)

        output = self.bert(content, attention_mask=masks, return_dict=True, output_attentions=True,
                           output_hidden_states=True)
        pooled = output['pooler_output']

        word_attention_gnn, semantic_attention_gnn = self.gnn(output, lengths)
        # print(torch.cosine_similarity(word_attention_gnn, semantic_attention_gnn, dim=-1))
        before_nodes_sim = torch.cosine_similarity(word_attention_gnn, semantic_attention_gnn, dim=-1)  # [N, L]
        # if not self.block_pooled:
        bert_out = self.bert_trans(pooled)
        word_attention_gnn_mix = self.word_interaction(word_attention_gnn, semantic_attention_gnn, masks)
        semantic_attention_gnn_mix = self.semantic_interaction(semantic_attention_gnn, word_attention_gnn, masks)
        mix_graphs_sim = torch.cosine_similarity(word_attention_gnn_mix, semantic_attention_gnn_mix, dim=-1)  # [N, L]

        word_attention_gnn = self.word_gate(bert_out, word_attention_gnn_mix)
        semantic_attention_gnn = self.semantic_gate(bert_out, semantic_attention_gnn_mix)
        after_graphs_sim = torch.cosine_similarity(word_attention_gnn, semantic_attention_gnn, dim=-1)  # [N]

        gnn_out = torch.cat((word_attention_gnn, semantic_attention_gnn), dim=1)
        if self.block_pooled:
            out = gnn_out
        else:
            out = torch.cat((bert_out, gnn_out), dim=1)
        # print(out.shape)

        out = self.dropout(out)
        out = self.fc(out)

        return out, [before_nodes_sim, mix_graphs_sim, after_graphs_sim]

    def evaluate(self, dataloader, phase='train'):
        self.criterion = torch.nn.CrossEntropyLoss()
        self.eval()
        all_predicted_result = []
        all_true_label = []
        all_sim_result = []

        with torch.no_grad():
            loss_list = []
            for idx, (contents, labels, lengths, masks, indexes) in enumerate(dataloader):
                if type(contents) == list:
                    contents = [content.to(self.device) for content in contents]
                else:
                    contents = contents.to(self.device)
                labels = labels.to(self.device)
                # lengths = lengths.to(self.device)
                masks = masks.to(self.device)

                model_result = self(contents, lengths, masks)

                if type(model_result) != torch.Tensor:
                    sim_result = model_result[-1]
                    model_result = model_result[0]
                loss = self.criterion(model_result, labels.long())  # 计算损失函数
                loss_list.append(loss.item())

                predicted_result = F.softmax(model_result, dim=1).detach().cpu().numpy()
                true_label = labels.cpu().numpy().tolist()

                all_predicted_result += predicted_result.tolist()
                all_true_label += true_label
                all_sim_result.append(sim_result)

            all_predicted_result = np.array(all_predicted_result)
            all_predicted_label = all_predicted_result.argmax(1)
            avg_loss = np.mean(loss_list)
            # print(len(list(zip(*all_sim_result))))
            all_sim_result = [torch.cat(sim, dim=0) for sim in zip(*all_sim_result)]

        if phase == 'benchmark':
            print(all_predicted_label)
            print(all_true_label)
        results = [avg_loss] + cal_metrics(all_true_label, all_predicted_result)

        return results, all_sim_result

    def test(self, test_dataloader, phase='test', epoch_log=None):
        # print('-' * 59)
        test_start_time = time.time()
        all_results, all_sim_result = self.evaluate(test_dataloader, phase)
        test_str = '-' * 59 + '\n' \
                   + '| end of test | time: {:5.2f}s |\n'.format(time.time() - test_start_time) \
                   + get_format_str(all_results, phase) \
                   + '-' * 59

        print(test_str)
        print([sim.shape for sim in all_sim_result])
        if epoch_log:
            epoch_log.write(test_str)

        return all_results, all_sim_result


class TopAttentionGNN(TopGNN):
    def __init__(self, num_head, dim_model, output_dim, keep_prob, device, top_rate=0.1, agg=None, reduce='mean',
                 k=10, alpha=0.2):
        super(TopAttentionGNN, self).__init__(num_head, dim_model, output_dim, keep_prob, device, top_rate, agg)
        self.word_gnn = TopAPPNPAttentionGNNModule(dim_model, keep_prob, self.device, top_rate, reduce, output_dim,
                                                   k, alpha)
        self.semantic_gnn = TopAPPNPAttentionGNNModule(dim_model, keep_prob, self.device, top_rate, reduce, output_dim,
                                                       k, alpha)
        self.hs_word_trans = nn.Linear(dim_model, dim_model)
        self.hs_semantic_trans = nn.Linear(dim_model, dim_model)

    def forward(self, output, lengths):
        # get the max attention matrices from different layers
        word_attention = torch.stack(output['attentions'][:3], dim=4).max(dim=4)[0].mean(dim=1)
        word_embed = self.hs_word_trans(torch.stack(output['hidden_states'][:3], dim=3).transpose(-2, -1)
                                        ).transpose(-2, -1).max(dim=3)[0]
        # semantic_attention = torch.stack(output['attentions'][-3:], dim=4).mean(dim=4)
        semantic_attention = torch.stack(output['attentions'][-3:], dim=4).max(dim=4)[0].mean(dim=1)
        semantic_embed = self.hs_semantic_trans(torch.stack(output['hidden_states'][-4:-1], dim=3).transpose(-2, -1)
                                                ).transpose(-2, -1).max(dim=3)[0]

        word_output = self.activation(self.word_fc(self.word_gnn(word_embed, word_attention, lengths)))
        semantic_output = self.activation(
            self.semantic_fc(self.semantic_gnn(semantic_embed, semantic_attention, lengths)))

        return word_output, semantic_output


class TopAPPNPAttentionGNNModule(TopAPPNPGNNModule):
    def __init__(self, dim_model, keep_prob, device, top_rate=0.1, reduce='mean', predict_dim=None, k=10, alpha=0.2):
        super(TopAPPNPAttentionGNNModule, self).__init__(dim_model, keep_prob, device, top_rate, reduce, predict_dim, k,
                                                         alpha)

    def forward(self, hidden_state, attention, lengths):
        batch_size = hidden_state.size(0)
        seq_len = hidden_state.size(1)
        # get the attention matrices from different heads
        # dealt_atten = attention.mean(dim=1)
        # dealt_atten = attention.max(dim=1)[0]
        dealt_atten = attention
        topk_result = torch.topk(dealt_atten, round(self.top_rate * hidden_state.size(1)))
        topk_values = topk_result.values
        topk_indices = topk_result.indices

        sub_graphs = [self.seq_to_graph(topk_values[i], topk_indices[i], hidden_state[i], self.reduce, lengths[i]) for i
                      in
                      range(batch_size)]
        batch_graph = dgl.batch(sub_graphs)

        result_node_embedding = self.conv(batch_graph, batch_graph.ndata['h'])

        batch_graph.ndata['h'] = result_node_embedding
        unbatched_graph = dgl.unbatch(batch_graph)
        # result_node_embedding = torch.cat([graph.ndata['h'].unsqueeze(dim=0) for graph in unbatched_graph], dim=0)
        result_node_embedding = torch.cat([
            torch.cat([graph.ndata['h'],
                       torch.zeros([(seq_len - graph.ndata['h'].shape[0]), self.predict_dim]).to(self.device)],
                      dim=0).unsqueeze(dim=0)
            for graph in unbatched_graph], dim=0)
        out = self.fc(self.dropout(self.activation(result_node_embedding)))
        out = self.ln(out)
        return out


class InteractionModule(nn.Module):
    def __init__(self, dim_model, attention_mode='normal'):
        '''
        combine the information of word and semantic
        '''
        super(InteractionModule, self).__init__()
        self.attention_mode = attention_mode
        self.dim_model = dim_model
        self.fc = nn.Linear(dim_model * 2, dim_model)
        self.output_trans = nn.Linear(dim_model, dim_model)
        self.context_trans = nn.Linear(dim_model, dim_model)
        self.activation = ACTIVATION()
        if self.attention_mode == 'biaffine':
            # self.bilinear = nn.Parameter(torch.empty(self.dim_model, self.dim_model))
            self.bilinear = nn.Linear(self.dim_model, self.dim_model, bias=False)
            # self.init_bilinear()
            self.U = nn.Linear(self.dim_model, 1)
            self.V = nn.Linear(self.dim_model, 1)

    # def init_bilinear(self):
    #     bound = 1 / math.sqrt(self.bilinear.shape[1])
    #     init.uniform_(self.bilinear, -bound, bound)

    def forward(self, output, context, attention_mask):
        lengths = torch.sum(attention_mask, dim=-1)
        batch_size = output.shape[0]
        context_len = output.shape[1]
        # attention mask
        key_padding_mask = torch.tensor([[0] * length + [-1e9] * (context_len - length) for length in lengths]
                                        ).unsqueeze(dim=1).to(DEVICE)
        if self.attention_mode == 'biaffine':
            # attn = torch.matmul(self.output_trans(output), self.bilinear)
            attn = self.bilinear(self.output_trans(output))
            attn = torch.bmm(attn, self.context_trans(context).transpose(1, 2))
            attn = attn + self.U(output).expand(attn.shape) + self.V(context).transpose(1, 2).expand(attn.shape)
            attn = attn + key_padding_mask
            attn = F.softmax(attn, dim=-1)
        else:
            # attn = self.activation(torch.bmm(self.output_trans(output), self.context_trans(context).transpose(1, 2)))
            attn = F.tanh(torch.bmm(self.output_trans(output), self.context_trans(context).transpose(1, 2)))
            attn = attn + key_padding_mask
            attn = F.softmax(attn.view(-1, context_len), dim=1).view(batch_size, -1, context_len)  # 对context内容的重新加权
        mix = torch.bmm(attn, context)
        combined = torch.cat((output, mix), dim=2)
        # output = self.activation(self.fc(combined.view(-1, 2 * self.dim_model))).view(batch_size, -1, self.dim_model)
        output = F.tanh(self.fc(combined.view(-1, 2 * self.dim_model))).view(batch_size, -1, self.dim_model)
        # output = output.mean(dim=1)
        output = (torch.sum((output.transpose(1, 2) * attention_mask.unsqueeze(dim=1)).transpose(1, 2),
                            dim=1).T / lengths).T
        # output = output.max(dim=1)[0]

        return output


class GateModule(nn.Module):
    def __init__(self, dim_model):
        '''
        let bert pooled filter the information
        '''
        super(GateModule, self).__init__()
        self.bert_trans = nn.Linear(dim_model, dim_model)
        self.gnn_trans = nn.Linear(dim_model, dim_model)
        self.activation = nn.Sigmoid()

    def forward(self, bert, gnn):
        alpha = self.activation(self.bert_trans(bert) + self.gnn_trans(gnn))
        return alpha * gnn


if __name__ == "__main__":
    def setup_seed(seed):
        # 设定随机种子
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
    model = DGCBERT(50000, 768, 2, 0, 512, model_path='../bert/base_bert/', mode='top_biaffine+softmax',
                    args=args).cuda()
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
