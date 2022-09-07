import argparse
import datetime

import torch

from models.base_model import BaseModel
from torch import nn
import torch.nn.functional as F
# from pytorch_pretrained_bert import BertModel
from transformers import BertModel, AutoModelForSequenceClassification


class BaseBert(BaseModel):
    def __init__(self, vocab_size, embed_dim, num_class, pad_index, pad_size=150, word2vec=None, keep_prob=0.5,
                 hidden_size=768, model_path=None, **kwargs):
        super(BaseBert, self).__init__(vocab_size, embed_dim, num_class, pad_index, pad_size, word2vec, keep_prob,
                                       model_path, **kwargs)
        self.model_name = 'BaseBERT'
        self.hidden_size = hidden_size
        self.num_head = 12
        self.head_dim = self.hidden_size // self.num_head
        assert self.hidden_size % self.num_head == 0

        self.bert = BertModel.from_pretrained(model_path, return_dict=True, output_attentions=True,
                                              output_hidden_states=True)
        self.adaptive_lr = True
        self.warmup = True
        # self.mode = 'normal'
        # print(self.bert)
        # self.bert = AutoModelForSequenceClassification.from_pretrained(model_path)
        print(model_path)
        # for name, param in self.bert.named_parameters():
        #     print(name, param.size())
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(self.hidden_size, self.num_class)

        self.dropout = nn.Dropout(keep_prob)

    def get_group_parameters(self, lr):
        print('use adaptive lr')
        bert_params = list(map(id, self.bert.parameters()))
        bert_params_list = [(n, p) for n, p in self.bert.named_parameters()]
        # for n, p in self.named_parameters():
        #     print(n)

        bert_dict = {}
        params_list = []
        bert_dict[-1] = list(filter(lambda value: 'embeddings.' in value[0], bert_params_list))
        for i in range(12):
            bert_dict[i] = list(filter(lambda value: '.' + str(i) + '.' in value[0], bert_params_list))
        bert_dict[11].extend(list(filter(lambda value: 'pooler.' in value[0], bert_params_list)))

        for i in range(-1, 12):
            gamma = 0.95 ** (11 - i)
            # print(i, gamma)
            current_value = bert_dict[i]
            current_list = [
                {
                    'params': [value[1] for value in current_value],
                    'lr': lr * gamma
                }
            ]
            params_list.extend(current_list)

        normal_list = filter(lambda p: id(p) not in bert_params, self.parameters())
        params_list.extend([
            {
                'params': [value for value in normal_list],
                'lr': lr * 10
            }
        ])

        return params_list

    def forward(self, content, lengths, masks, **kwargs):
        content = content.permute(1, 0)

        output = self.bert(content, attention_mask=masks)

        out = output['pooler_output']

        out = self.dropout(out)
        out = self.fc(out)
        # out = self.fc2(out)
        # out = F.softmax(out, dim=1)

        return out


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()
    input_args = args.__dict__
    input_args['predict_dim'] = 256
    model = BaseBert(50000, 768, 2, 0, 512, model_path='../bert/base_bert/', mode='adap', args=input_args).cuda()
    x = torch.zeros(256, 2, dtype=torch.int).cuda()
    print(model(x, torch.ones(2, dtype=torch.int).cuda(), torch.zeros(2, 256, dtype=torch.int).cuda()))
    # model.get_group_parameters(0.1)

    if args.phase == 'test':
        print('This is a test process.')
    else:
        print('error! No such method!')
    end_time = datetime.datetime.now()
    print('{} takes {} seconds'.format(args.phase, (end_time - start_time).seconds))

    print('Done base_model!')
