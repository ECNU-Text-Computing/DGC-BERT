import argparse
import datetime
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

from pretrained_models.BAG import *
from torch import nn
import torch.nn.functional as F
from transformers import BertModel


class BAGT(BAG):
    def __init__(self, vocab_size, embed_dim, num_class, pad_index, word2vec=None, keep_prob=0.5, pad_size=150,
                 hidden_size=768, model_path=None, mode=None, model_type='BERT', **kwargs):
        super(BAGT, self).__init__(vocab_size, embed_dim, num_class, pad_index, word2vec, keep_prob, pad_size,
                                   hidden_size, model_path, mode, model_type, **kwargs)
        self.model_name = 'BAGT'
        self.block_pooled = False
        self.reduce_method = 'mean'
        if mode:
            self.model_name = self.model_name + '_' + mode + '_' + model_type
            print('using ' + mode)
        else:
            self.model_name = self.model_name + '_' + model_type
        # print('top rate', TOP_RATE)
        if mode == 'top':
            self.gnn = TopGNN(self.num_head, self.hidden_size, self.predict_dim, keep_prob, self.device, TOP_RATE,
                              'base', self.reduce_method)
        elif mode == 'top_sage':
            self.gnn = TopGNN(self.num_head, self.hidden_size, self.predict_dim, keep_prob, self.device, TOP_RATE,
                              'SAGE', self.reduce_method)
        elif mode == 'top_appnp':
            self.gnn = TopGNN(self.num_head, self.hidden_size, self.predict_dim, keep_prob, self.device, TOP_RATE,
                              'APPNP', self.reduce_method)
        elif mode == 'top_tag':
            self.gnn = TopGNN(self.num_head, self.hidden_size, self.predict_dim, keep_prob, self.device, TOP_RATE,
                              'TAG', self.reduce_method)
        else:
            self.gnn = BaseGNN(self.num_head, self.hidden_size, self.predict_dim, keep_prob)

        self.loss_param = nn.Parameter(torch.tensor(1, dtype=torch.float))
        self.final_dim = 2 * self.predict_dim if self.block_pooled else 3 * self.predict_dim
        self.bert_trans = nn.Sequential(
            # nn.Dropout(p=keep_prob),
            FNN(self.hidden_size, keep_prob),
            nn.Tanh(),
            nn.Linear(int(self.hidden_size/4), self.predict_dim),
            nn.Tanh()
        )
        # print(self.bert_trans)
        self.fc = nn.Sequential(
            FNN(self.final_dim, keep_prob),
            # nn.ReLU(),
            nn.Tanh(),
            # nn.Dropout(p=keep_prob),
            nn.Linear(int(self.final_dim / 4), self.num_class),
        )
    def forward(self, content, lengths, masks, **kwargs):
        content = content.permute(1, 0)

        output = self.bert(content, attention_mask=masks, return_dict=True, output_attentions=True,
                           output_hidden_states=True)
        pooled = output['pooler_output']

        word_attention_gnn, semantic_attention_gnn = self.gnn(output)
        if not self.block_pooled:
            bert_out = self.bert_trans(pooled)

        gnn_out = torch.cat((word_attention_gnn, semantic_attention_gnn), dim=1)
        if self.block_pooled:
            out = gnn_out
        else:
            out = torch.cat((bert_out, gnn_out), dim=1)

        out = self.dropout(out)
        out = self.fc(out)

        return out

    def train_model(self, dataloader, epoch, criterion, optimizer, epoch_log):
        """
        stop training BERT after 3 epochs
        """
        if epoch > 3:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.train()
        total_acc, total_count = 0, 0
        log_interval = 10
        start_time = time.time()
        loss_list = []
        all_predicted_result = []
        all_true_label = []

        for idx, (contents, labels, lengths, masks, indexes) in enumerate(dataloader):
            start_time = time.time()
            if type(contents) == list:
                contents = [content.to(self.device) for content in contents]
            else:
                contents = contents.to(self.device)
            # print('io-time:', time.time()-start_time)
            labels = labels.to(self.device)
            lengths = lengths.to(self.device)
            masks = masks.to(self.device)
            # print('io-time:', time.time() - start_time)

            optimizer.zero_grad()
            predicted_result = self(contents, lengths, masks)
            # print('forward-time:', time.time() - start_time)
            loss = criterion(predicted_result, labels.long())
            # print('loss-time:', time.time() - start_time)
            loss.backward()
            # print('backforward-time:', time.time() - start_time)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
            if self.warmup:
                self.scheduler.step()
            optimizer.step()
            # print('optimizer-time:', time.time() - start_time)
            all_predicted_result += F.softmax(predicted_result, dim=1).detach().cpu().numpy().tolist()
            all_true_label += labels.cpu().numpy().tolist()
            total_acc += (predicted_result.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
            loss_list.append(loss.item())
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| accuracy {:8.3f} | loss {:8.3f}'.format(epoch, idx, len(dataloader),
                                                                 total_acc / total_count, loss.item()))
                epoch_log.write('| epoch {:3d} | {:5d}/{:5d} batches '
                                '| accuracy {:8.3f} | loss {:8.3f}\n'.format(epoch, idx, len(dataloader),
                                                                             total_acc / total_count, loss.item()))
                total_acc, total_count = 0, 0
                start_time = time.time()
            # print('val-time:', time.time() - start_time)

        all_predicted_result = np.array(all_predicted_result)
        all_predicted_label = all_predicted_result.argmax(1)

        acc = accuracy_score(all_true_label, all_predicted_label)
        prec = precision_score(all_true_label, all_predicted_label)
        recall = recall_score(all_true_label, all_predicted_label)
        f1 = f1_score(all_true_label, all_predicted_label, average='binary')
        maf1 = f1_score(all_true_label, all_predicted_label, average='macro')
        # mif1 = f1_score(all_true_label, all_predicted_label, average='micro')s
        auc = roc_auc_score(all_true_label, all_predicted_result[:, 1])
        log_loss_value = log_loss(all_true_label, all_predicted_result)
        avg_loss = np.mean(loss_list)

        print(
            '-' * 59 + '\n' +
            '| average loss {:4.3f} | train accuracy {:8.3f} |\n'
            '| precision {:8.3f} | recall {:10.3f} |\n'
            '| macro-f1 {:9.3f} | normal-f1 {:7.3f} |\n'
            '| auc {:14.3f} | log_loss {:8.3f} |\n'.format(avg_loss, acc, prec, recall, maf1, f1, auc, log_loss_value)
        )
        epoch_log.write(
            '-' * 59 + '\n' +
            '| average loss {:4.3f} | train accuracy {:8.3f} |\n'
            '| precision {:8.3f} | recall {:10.3f} |\n'
            '| macro-f1 {:9.3f} | normal-f1 {:7.3f} |\n'
            '| auc {:14.3f} | log_loss {:8.3f} |\n'.format(avg_loss, acc, prec, recall, maf1, f1, auc, log_loss_value)
        )

        return [avg_loss, acc, prec, recall, maf1, f1, auc, log_loss_value]


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
    model = BAGT(50000, 768, 2, 0, 512, model_path='../bert/base_bert/', mode='top_appnp').cuda()
    seq_len = 128
    bs = 2
    # x = torch.randint(0, 20000, (seq_len, 2)).cuda()
    # x = torch.ones((seq_len, 2)).cuda()
    # x = torch.zeros(256, 2, dtype=torch.int).cuda()
    x = torch.tensor(([list(range(seq_len * i, seq_len * (i + 1))) for i in range(bs)]), dtype=torch.int).permute(1,
                                                                                                                  0).cuda()
    print(x.shape)
    result = model(x, torch.ones(bs, dtype=torch.int).cuda(), torch.zeros(bs, seq_len, dtype=torch.int).cuda())
    print(result[0])
    print(result[1].shape, result[2].shape)
    word_result, semantic_result = result[1], result[2]