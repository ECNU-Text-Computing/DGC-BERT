import argparse
import datetime

import torch
from torch import nn

from pretrained_models.base_bert import BaseBert


class SciBERT(BaseBert):
    def __init__(self, vocab_size, embed_dim, num_class, pad_index, word2vec=None, keep_prob=0.5, pad_size=150,
                 hidden_size=768, model_path=None, **kwargs):
        super(SciBERT, self).__init__(vocab_size, embed_dim, num_class, pad_index, pad_size, word2vec, keep_prob,
                                      hidden_size, model_path, **kwargs)
        self.model_name = 'SciBERT'
        # if kwargs['mode']:
        #     self.model_name += '_' + kwargs['mode']
        #     print(self.model_name)
        self.adaptive_lr = True

        print(self.fc)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()
    model = SciBERT(50000, 768, 2, 0, 512, model_path='../bert/base_bert/', mode='pro').cuda()
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
