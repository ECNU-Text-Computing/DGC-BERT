import argparse
import datetime

import numpy as np
import pandas as pd
import torch
from torchtext.vocab import build_vocab_from_iterator

# from clean_latex import extract
import json
import random

import torchtext
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

UNK, PAD, CLS, SEP = '[UNK]', '[PAD]', '[CLS]', '[SEP]'
tokenizer = get_tokenizer('basic_english')
parts = ['title', 'abstract', 'intro', 'related', 'methods', 'conclusion', 'authors']
text_parts = ['title', 'abstract', 'intro', 'related', 'methods', 'conclusion']
phases = ['train', 'val', 'test']


def split_data(in_path, out_path, shuffle=True, rate=0.7, seed=123):
    # with open(path + '/' + 'label_map.json', 'r') as fp:
    #     load_map_dict = json.load(fp)
    all_data = json.load(open(in_path, 'r'))
    data = [{key: all_data[key][i] for key in all_data.keys()} for i in range(len(all_data['abstract']))]
    print('original data length', len(data))
    # > 20
    selected_data = list(filter(lambda x: len(tokenizer('. '.join(x['abstract']).strip())) > 20, data))
    print('selected data length', len(selected_data))
    data_keys = all_data.keys()
    all_data = {key: [] for key in data_keys}
    for paper in selected_data:
        for key in data_keys:
            all_data[key].append(paper[key])

    all_labels = all_data['label']
    all_contents = [all_data[part] for part in parts]
    all_contents = list(zip(*all_contents))
    all_indexes = list(range(len(all_contents)))
    print(len(all_contents))

    if shuffle:
        data = list(zip(all_contents, all_labels, all_indexes))
        random.seed(seed)
        random.shuffle(data)
        all_contents, all_labels, all_indexes = zip(*data)

    total_count = len(all_contents)

    train_contents = all_contents[:int(total_count * rate)]
    train_labels = all_labels[:int(total_count * rate)]
    train_indexes = all_indexes[:int(total_count * rate)]

    val_contents = all_contents[int(total_count * rate): int(total_count * ((1 - rate) / 2 + rate))]
    val_labels = all_labels[int(total_count * rate): int(total_count * ((1 - rate) / 2 + rate))]
    val_indexes = all_indexes[int(total_count * rate): int(total_count * ((1 - rate) / 2 + rate))]

    test_contents = all_contents[int(total_count * ((1 - rate) / 2 + rate)):]
    test_labels = all_labels[int(total_count * ((1 - rate) / 2 + rate)):]
    test_indexes = all_indexes[int(total_count * ((1 - rate) / 2 + rate)):]

    contents = [train_contents, val_contents, test_contents]
    labels = [train_labels, val_labels, test_labels]
    indexes = [train_indexes, val_indexes, test_indexes]

    for phase in phases:
        with open(out_path + '/' + '{}_label_{}'.format(phase, seed), 'w') as fw:
            json.dump(labels[phases.index(phase)], fw)
        with open(out_path + '/' + '{}_index_{}'.format(phase, seed), 'w') as fw:
            json.dump(indexes[phases.index(phase)], fw)
        content = contents[phases.index(phase)]
        print(len(content))
        content = list(zip(*content))
        print(len(content))
        for part in parts:
            with open(out_path + '/' + '{}_{}_{}'.format(phase, part, seed),
                      'w') as fw:
                json.dump(content[parts.index(part)], fw)


def get_vocab(path, seed=123, word_min_freq=5):
    all_dict = {}
    text_contents = []
    for part in parts:
        all_dict[part] = json.load(open(path + '/' + 'train_{}_{}'.format(part, seed), 'r'))
    for part in text_parts:
        print(part)
        if part == 'title':
            text_contents.extend(all_dict[part])
        else:
            text_contents.extend([' '.join(line) for line in all_dict[part]])
    print('buliding text vocab...')
    vocab = build_vocab_from_iterator(yield_tokens(text_contents), specials=[UNK, PAD], min_freq=word_min_freq)
    vocab.set_default_index(vocab[UNK])
    torch.save(vocab, path + '/' + 'vocab_{}.pth'.format(seed))
    print(len(vocab.vocab))
    authors = all_dict['authors']
    print('buliding author vocab...')
    authors_vocab = build_vocab_from_iterator(yield_authors(authors), specials=[UNK, PAD], min_freq=2)
    authors_vocab.set_default_index(authors_vocab[UNK])
    torch.save(authors_vocab, path + '/' + 'authors_vocab_{}.pth'.format(seed))
    print(len(authors_vocab.vocab))


def yield_tokens(data_iter):
    for content in data_iter:
        yield tokenizer(content.encode('utf-8', 'replace').decode('utf-8'))
        # yield tokenizer(content)


def yield_authors(data_iter):
    for authors in data_iter:
        yield authors.split(',')


def make_data(in_path, out_path, seed, rate):
    split_data(in_path, out_path, seed=seed, rate=rate)
    get_vocab(out_path, seed)


def make_chunk_data(in_path, out_path, seed, bert_path):
    # phases = ['test']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    bert_model = BertModel.from_pretrained(bert_path, return_dict=True, output_hidden_states=True).to(device)
    parts = ['title', 'abstract', 'intro', 'related', 'methods', 'conclusion']
    cls, sep = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
    result_dict = {}
    for phase in phases:
        indexes = json.load(open(in_path + '{}_index_{}'.format(phase, seed)))
        labels = json.load(open(in_path + '{}_label_{}'.format(phase, seed)))
        result_dict[phase] ={
            'text': [],
            'labels': labels,
            'lengths': [],
            'indexes': indexes,
            'masks': []
        }
        temp_data = {part: json.load(open(in_path + '{}_{}_{}'.format(phase, part, seed))) for part in parts}
        for i in tqdm(range(len(indexes))):
            temp_result = []
            temp_content = '. '.join(['. '.join(temp_data[part][i]) for part in parts])
            tokens = tokenizer(temp_content)['input_ids']
            start_list = np.arange(0, len(tokens), 460)
            for start_idx in start_list:
                cur_tokens = torch.tensor([cls] + tokens[start_idx: start_idx+510] + [sep]).unsqueeze(dim=0).to(device)
                # print(len(cur_tokens))
                mask = torch.ones(cur_tokens.shape).to(device)
                # print(mask.shape)
                output = bert_model(cur_tokens, attention_mask=mask)
                node_embedding = output['hidden_states'][-1][0].mean(dim=0, keepdim=True).detach().cpu()
                temp_result.append(node_embedding)
            # print(torch.cat(temp_result, dim=0).shape)
            result_dict[phase]['text'].append(torch.cat(temp_result, dim=0))
            result_dict[phase]['lengths'].append(len(temp_result))
            result_dict[phase]['masks'].append(0)

    torch.save(result_dict, out_path + 'chunk_data_{}'.format(seed))




if __name__ == '__main__':
    # split_data('.')
    # get_vocab('.')
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')
    parser.add_argument('--in_path', default=None, help='the input.')
    parser.add_argument('--out_path', default=None, help='the output.')
    parser.add_argument('--seed', default=123, help='the seed.')

    args = parser.parse_args()

    if args.phase == 'test':
        print('This is a test process.')
        # data = json.load(open('../data/AAPR_full/test_intro_123'))
        # print(len(data))
        # print(data[0])
        make_chunk_data('../data/PeerRead_full/', out_path='../data/PeerRead_full/', seed=333, bert_path='../bert/base_bert/')
    elif args.phase == 'make_data':
        in_path = args.in_path
        out_path = args.out_path
        seed = args.seed
        split_data(in_path, out_path, seed=seed)
        get_vocab(out_path, seed)

    # all_data = json.load(open('../data/AAPR_full/dealt_data', 'r'))

    # tokenizer = get_tokenizer('basic_english')
    # part = 'related'
    # len_list = []
    # tokens_list = []
    # special_list = []
    # length = len(all_data[part])
    # print(length)
    # print(len([line for line in all_data[part] if ''.join(line).strip()=='']))
    # temp = [[sent for sent in line if len(tokenizer(sent))>=5] for line in all_data[part]]
    # for i in range(length):
    #     len_list.append(len(temp[i]))
    #     for tokens in temp[i]:
    #         tokens_len = len(tokenizer(tokens))
    #         tokens_list.append(tokens_len)
    #         if tokens_len > 100:
    #             special_list.append(tokens)
    # # print(len([line for line in len_list if line > 1]))
    # print(pd.DataFrame(len_list).describe(percentiles=[0.1, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]))
    # print(pd.DataFrame(tokens_list).describe(percentiles=[0.1, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]))
    # print('\n'.join(special_list[:5]))
