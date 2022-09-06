import argparse
import json

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.vocab import vocab
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UNK, PAD, CLS, SEP = '[UNK]', '[PAD]', '[CLS]', '[SEP]'


class SelectedMethod(nn.Module):
    def __init__(self, method='last2'):
        super(SelectedMethod, self).__init__()
        self.method = method

    def forward(self, x):
        if self.method == 'last2':
            result = x[-2]
        elif self.method == 'cls':
            result = x[-2][0]
        elif self.method == 'last':
            result = x[-1]
        elif self.method == 'fnl':
            result = torch.cat((x[1],x[-1]), dim=0)
        else:
            result = x
        return result


class BertTrans(nn.Module):
    def __init__(self, model_path, method='last2'):
        super(BertTrans, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.selected_method = SelectedMethod(method)

    def forward(self, content, masks):
        content = content.permute(1, 0)
        output = self.bert(content, attention_mask=masks, return_dict=True, output_attentions=True,
                           output_hidden_states=True)
        output = self.selected_method(output['hidden_states'])

        return output


class DataTrans:
    def __init__(self, data_source='openreview', pad_size=256, seed=123, tokenizer_path=None):
        self.data_source = data_source
        self.pad_size = pad_size
        self.seed = seed
        self.data_root = './data/'
        self.data_cat_path = self.data_root + self.data_source + '/'
        self.tokenizer_path = tokenizer_path

    def load_data(self):
        with open(self.data_root + self.data_source + '/' + 'all_contents.list', 'r') as fp:
            all_contents = fp.readlines()
        all_indexes = list(range(len(all_contents)))

        # with open(self.data_root + self.data_source + '/' + 'all_labels.list', 'r') as fp:
        #     all_labels = fp.readlines()
        return all_contents, all_indexes

    def build_vocab(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)
        self.vocab = vocab(self.tokenizer.vocab, min_freq=0)
        return self.vocab

    def get_dataloader(self, contents, indexes, batch_size):
        self.text_pipeline = lambda x: self.get_dealt_text(x, self.pad_size, True)
        dataloader = DataLoader(list(zip(contents, indexes)), batch_size=batch_size,
                                shuffle=False, collate_fn=self.collate_batch)
        return dataloader

    def get_dealt_text(self, text, pad_size=None, using_BERT=True):
        mask = []
        if using_BERT:
            tokens = self.tokenizer.tokenize(text.strip())
            tokens = [CLS] + tokens
            seq_len = len(tokens)
        else:
            tokens = self.tokenizer(text.strip())
            seq_len = len(tokens)
        if pad_size:
            if len(tokens) < pad_size:
                tokens.extend([PAD] * (pad_size - len(tokens)))
                mask = [1] * len(tokens) + [0] * (pad_size - len(tokens))
            else:
                tokens = tokens[:pad_size]
                seq_len = pad_size
                mask = [1] * pad_size
        return torch.tensor(self.vocab(tokens), dtype=torch.int64), seq_len, mask

    def collate_batch(self, batch):
        # 具体处理每条数据的方法，使用词典构建的流水线进行处理，转换成词向量，同时等长以转换成张量
        content_list = []
        length_list = []
        mask_list = []
        index_list = []
        PAD_IDX = self.vocab[PAD]
        for (_content, _index) in batch:
            processed_content, seq_len, mask = self.text_pipeline(_content)
            content_list.append(processed_content)
            length_list.append(seq_len)
            mask_list.append(mask)
            index_list.append(_index)
        # content_list = torch.cat(content_list)
        # 固定长度转换为张量
        content_batch = pad_sequence(content_list, padding_value=PAD_IDX)
        length_list = torch.tensor(length_list, dtype=torch.int64)
        mask_list = torch.tensor(mask_list, dtype=torch.int8)
        # index_list = torch.tensor(index_list, dtype=torch.int64)
        # print(len(label_list))
        return content_batch.to(device), length_list.to(device), mask_list.to(device), index_list


def main(data_source, pad_size, bert_dict, batch_size, cut_batch=300):
    dataTrans = DataTrans(data_source, pad_size, tokenizer_path=bert_dict['vocab_path'])
    all_contents, all_indexes = dataTrans.load_data()
    dataTrans.build_vocab()
    dataloader = dataTrans.get_dataloader(all_contents, all_indexes, batch_size)

    trans_model = BertTrans(bert_dict['model_path']).to(device)
    result_dict = {}
    count = 0

    for idx, (contents, lengths, masks, indexes) in enumerate(dataloader):
        contents.requires_grad = False
        converted_result = trans_model(contents, masks)
        indexes = list(indexes)
        for i in range(len(indexes)):
            result_dict[indexes[i]] = {
                'matrix': converted_result[i].detach().cpu(),
                'lens': lengths[i].item()
            }
        if ((idx > 0) & (idx % cut_batch == 0)) | (idx == len(dataloader) - 1):
            torch.save(result_dict, dataTrans.data_cat_path + 'bert_embed_{}'.format(count))
            result_dict = {}
            count += 1
        # if count == 2:
        #     break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='main', help='the function name.')
    parser.add_argument('--data_source', default='AAPR', help='the data source.')
    parser.add_argument('--pad_size', default=256, help='the data source.')
    parser.add_argument('--type', default='BERT', help='the model type.')
    parser.add_argument('--batch_size', default=32, help='the model type.')
    parser.add_argument('--cut_batch', default=300, help='the model type.')

    args = parser.parse_args()

    if args.phase == 'test':
        print('This is a test process.')
        sample_config = {
            'model_path': './bert/base_bert/',
            'vocab_path': './bert/base_bert/bert-base-uncased-vocab.txt'
        }
        main('AAPR', 256, sample_config, 2, 10)

    elif args.phase == 'main':
        bert_config = json.load(open('./configs/pretrained_types.json', 'r'))
        bert_dict = bert_config[args.type]
        main(args.data_source, int(args.pad_size), bert_dict, int(args.batch_size), int(args.cut_batch))
