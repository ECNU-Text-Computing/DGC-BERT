import argparse
import codecs
import datetime
import os
import string
import sys
from collections import OrderedDict, Counter

sys.path.insert(0, './')
sys.path.insert(0, '../')
# print(sys.path)
from scipy.sparse import coo_matrix
from transformers import BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec, LdaModel, TfidfModel
from gensim.corpora.dictionary import Dictionary
from torch.nn.utils.rnn import pad_sequence
from scripts.aapr_full import make_data, make_chunk_data
from scripts.peerread_full import make_pr_data
from tqdm import tqdm

import json
import random
import numpy as np
from joblib import dump
import pandas as pd
import re

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import Vectors
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
puncs = string.punctuation
punc_trans = {i: ' ' for i in puncs}
punc_table = str.maketrans(punc_trans)

tokenizer = get_tokenizer('basic_english')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UNK, PAD, CLS, SEP = '[UNK]', '[PAD]', '[CLS]', '[SEP]'
REVIEWS_CUT = ' __!reviews!__ '


def get_vote_split(data_lists):
    dealt_data_lists = []
    for data_pair in data_lists:
        dealt_content = []
        dealt_label = []
        dealt_index = []
        index = 0
        for (contents, label) in zip(data_pair[0], data_pair[1]):
            split_contents = contents.split(REVIEWS_CUT)
            review_count = len(split_contents)
            split_labels = [label] * review_count
            split_indexes = [index] * review_count

            dealt_content += split_contents
            dealt_label += split_labels
            dealt_index += split_indexes
            index += 1
        dealt_data_lists.append([dealt_content, dealt_label, dealt_index])
    return dealt_data_lists


class DataProcessor(object):
    def __init__(self, data_source='AAPR', pad_size=1500, seed=123):
        print('Init...')
        self.data_root = './data/'
        self.data_source = data_source
        self.seed = int(seed)
        if self.data_source == 'AAPR':
            self.data_cat_path = self.data_root + self.data_source + '/'
        elif self.data_source == 'AAPR_full':
            self.data_cat_path = self.data_root + self.data_source + '/'
        elif self.data_source == 'PeerRead':
            self.data_cat_path = self.data_root + self.data_source + '/'
        elif self.data_source == 'PeerRead_full':
            self.data_cat_path = self.data_root + self.data_source + '/'
        else:
            self.data_cat_path = self.data_root + self.data_source + '/'

        self.tokenizer = tokenizer
        self.pad_size = pad_size
        self.vocab = None
        self.vectors = None
        self.text_pipeline = None
        self.label_pipeline = None

    def show_openreview_cat(self):
        with open(self.data_cat_path, 'r') as f:
            line = f.readline().strip()
            print(line)

    def extract_data(self):
        content_list, label_list = [], []
        fw_content = open(self.data_root + self.data_source + '/' + 'content.list', 'w')
        fw_label = open(self.data_root + self.data_source + '/' + 'label.list', 'w')
        if self.data_source == 'AAPR':
            data = json.load(open(self.data_cat_path + 'dealt_data', 'r'))
            # data = pd.DataFrame(data)
            # print(data.head())
            data = [{key: data[key][i] for key in data.keys()} for i in range(len(data['abstract']))]
            print('original data length', len(data))
            # filter abstract length > 20
            data = list(filter(lambda x: len(self.tokenizer('. '.join(x['abstract']).strip())) > 20, data))
            # data['length'] = data['abstract'].apply(lambda x: len(self.tokenizer('. '.join(x).strip())) > 20)
            # data = data[data['length']]
            data = pd.DataFrame(data)
            print(data.head())
            label_lines = data['label'].to_list()
            abstract_lines = data['abstract'].to_list()
            title_lines = data['title'].to_list()
            # content_lines = [' '.join(line) for line in content_lines]
            content_lines = []
            for i in range(len(title_lines)):
                content_lines.append(title_lines[i] + '. ' + '. '.join(abstract_lines[i]))
                # content_lines.append('. '.join(abstract_lines[i]))
        elif self.data_source == 'PeerRead':
            datasets = ['arxiv.cs.ai_2007-2017', 'arxiv.cs.cl_2007-2017', 'arxiv.cs.lg_2007-2017']
            result_list = []
            # filter abstract length > 20
            for dataset in datasets:
                result_list.append(json.load(open(self.data_cat_path + '{}.json'.format(dataset), 'r')))
            phases = ['train', 'dev', 'test']
            result_dict = {phase: {} for phase in phases}
            content_lines = []
            label_lines = []
            start_index = 0
            for phase in phases:
                phase_name = 'val' if phase == 'dev' else phase
                phase_result = []
                for i in range(len(datasets)):
                    phase_result.extend(result_list[i][phase])
                df = pd.DataFrame(phase_result)
                print('original data length', len(df))
                df = df[df['title'].notna()]
                df = df[df['abstractText'].notna()]
                df['length'] = df['abstractText'].apply(lambda x: len(self.tokenizer(x.strip())) > 20)
                df = df[df['length']]
                df['content'] = df['title'] + '. ' + df['abstractText']
                # df['content'] = df['abstractText']
                result_dict[phase]['content_lines'] = df['content'].astype(str).to_list()
                result_dict[phase]['label_lines'] = df['accepted'].astype(bool).astype(int).astype(str).to_list()
                content_lines.extend(result_dict[phase]['content_lines'])
                label_lines.extend(result_dict[phase]['label_lines'])
                with open(self.data_root + self.data_source + '/' + '{}_contents.list'.format(phase_name), 'w+') as fw:
                    fw.writelines([' '.join(re.split(r'[\s]+', content.strip())) + '\n'
                                   for content in result_dict[phase]['content_lines']])
                with open(self.data_root + self.data_source + '/' + '{}_labels.list'.format(phase_name), 'w+') as fw:
                    fw.writelines([label + '\n' for label in result_dict[phase]['label_lines']])
                with open(self.data_root + self.data_source + '/' + '{}_indexes.list'.format(phase_name), 'w+') as fw:
                    fw.writelines([str(index) + '\n' for index in
                                   range(start_index, start_index + len(result_dict[phase]['content_lines']))])
                start_index += len(result_dict[phase]['content_lines'])
                print('dealt data length', len(result_dict[phase]['content_lines']))
                print(start_index)
        else:
            print('no this data source')
            raise Exception

        print('dealt data length', len(content_lines))
        for i in range(len(content_lines)):
            # remove blanks
            content = ' '.join(re.split(r'[\s]+', content_lines[i].strip()))
            fw_content.write(content + '\n')
            fw_label.write(label_lines[i] + '\n')
            content_list.append(content)
            label_list.append(label_lines[i])

        fw_content.close()
        fw_label.close()

        return content_list, label_list

    def label_map(self):
        label_map_dict = {}
        with open(self.data_root + self.data_source + '/' + 'label.list') as fp:
            all_lines = fp.readlines()
            # all_lines_set = set(all_lines)
            all_lines_list = list(set(all_lines))
            all_lines_list.sort()
            # all_lines_list.reverse()
            for label in all_lines_list:
                if label not in label_map_dict:
                    label_map_dict[label.strip()] = len(label_map_dict)
        print(label_map_dict)
        with open(self.data_root + self.data_source + '/' + 'label_map.json', 'w') as fw:
            json.dump(label_map_dict, fw, ensure_ascii=False)
        self.num_class = len(label_map_dict)
        return label_map_dict

    def split_data(self, shuffle=True, rate=0.7, fixed_num=None):
        print('split_data')
        with open(self.data_root + self.data_source + '/' + 'label_map.json', 'r') as fp:
            load_map_dict = json.load(fp)

        with open(self.data_root + self.data_source + '/' + 'content.list', 'r') as f_content:
            all_contents = list(map(lambda x: x.strip(), f_content.readlines()))

        with open(self.data_root + self.data_source + '/' + 'label.list', 'r') as f_label:
            all_labels = list(map(lambda x: int(load_map_dict[x.strip()]), f_label.readlines()))

        all_indexes = list(range(len(all_contents)))

        if shuffle:
            data = list(zip(all_contents, all_labels, all_indexes))
            random.seed(self.seed)
            print('data_processor seed', self.seed)
            # print(random.getstate())
            random.shuffle(data)
            all_contents, all_labels, all_indexes = zip(*data)

        total_count = len(all_contents)

        if not fixed_num:
            train_contents = all_contents[:int(total_count * rate)]
            train_labels = all_labels[:int(total_count * rate)]
            train_indexes = all_indexes[:int(total_count * rate)]

            val_contents = all_contents[int(total_count * rate): int(total_count * ((1 - rate) / 2 + rate))]
            val_labels = all_labels[int(total_count * rate): int(total_count * ((1 - rate) / 2 + rate))]
            val_indexes = all_indexes[int(total_count * rate): int(total_count * ((1 - rate) / 2 + rate))]

            test_contents = all_contents[int(total_count * ((1 - rate) / 2 + rate)):]
            test_labels = all_labels[int(total_count * ((1 - rate) / 2 + rate)):]
            test_indexes = all_indexes[int(total_count * ((1 - rate) / 2 + rate)):]
        else:
            train_contents = all_contents[:fixed_num[0]]
            train_labels = all_labels[:fixed_num[0]]
            train_indexes = all_indexes[:fixed_num[0]]

            val_contents = all_contents[fixed_num[0]: fixed_num[0] + fixed_num[1]]
            val_labels = all_labels[fixed_num[0]: fixed_num[0] + fixed_num[1]]
            val_indexes = all_indexes[fixed_num[0]: fixed_num[0] + fixed_num[1]]

            test_contents = all_contents[fixed_num[0] + fixed_num[1]:fixed_num[0] + fixed_num[1] + fixed_num[2]]
            test_labels = all_labels[fixed_num[0] + fixed_num[1]:fixed_num[0] + fixed_num[1] + fixed_num[2]]
            test_indexes = all_indexes[fixed_num[0] + fixed_num[1]:fixed_num[0] + fixed_num[1] + fixed_num[2]]

        with open(self.data_root + self.data_source + '/' + 'all_contents.list', 'w') as fw:
            for line in all_contents:
                fw.write(line + '\n')

        with open(self.data_root + self.data_source + '/' + 'all_labels.list', 'w') as fw:
            for line in all_labels:
                fw.write(str(line) + '\n')

        with open(self.data_root + self.data_source + '/' + 'all_indexes.list', 'w') as fw:
            for line in all_indexes:
                fw.write(str(line) + '\n')

        with open(self.data_root + self.data_source + '/' + 'train_contents.list', 'w') as fw:
            for line in train_contents:
                fw.write(line + '\n')

        with open(self.data_root + self.data_source + '/' + 'train_labels.list', 'w') as fw:
            for line in train_labels:
                fw.write(str(line) + '\n')

        with open(self.data_root + self.data_source + '/' + 'train_indexes.list', 'w') as fw:
            for line in train_indexes:
                fw.write(str(line) + '\n')

        with open(self.data_root + self.data_source + '/' + 'val_contents.list', 'w') as fw:
            for line in val_contents:
                fw.write(line + '\n')

        with open(self.data_root + self.data_source + '/' + 'val_labels.list', 'w') as fw:
            for line in val_labels:
                fw.write(str(line) + '\n')

        with open(self.data_root + self.data_source + '/' + 'val_indexes.list', 'w') as fw:
            for line in val_indexes:
                fw.write(str(line) + '\n')

        with open(self.data_root + self.data_source + '/' + 'test_contents.list', 'w') as fw:
            for line in test_contents:
                fw.write(line + '\n')

        with open(self.data_root + self.data_source + '/' + 'test_labels.list', 'w') as fw:
            for line in test_labels:
                fw.write(str(line) + '\n')

        with open(self.data_root + self.data_source + '/' + 'test_indexes.list', 'w') as fw:
            for line in test_indexes:
                fw.write(str(line) + '\n')

    def load_data(self, data_source=None, seed=None):
        if not data_source:
            data_source = self.data_source
        with open(self.data_root + data_source + '/' + 'all_contents.list', 'r') as fp:
            all_contents = fp.readlines()

        with open(self.data_root + data_source + '/' + 'all_labels.list', 'r') as fp:
            all_labels = fp.readlines()

        with open(self.data_root + data_source + '/' + 'train_contents.list', 'r') as fp:
            train_contents = fp.readlines()

        with open(self.data_root + data_source + '/' + 'train_labels.list', 'r') as fp:
            train_labels = fp.readlines()

        with open(self.data_root + data_source + '/' + 'train_indexes.list', 'r') as fp:
            train_indexes = fp.readlines()

        with open(self.data_root + data_source + '/' + 'val_contents.list', 'r') as fp:
            val_contents = fp.readlines()

        with open(self.data_root + data_source + '/' + 'val_labels.list', 'r') as fp:
            val_labels = fp.readlines()

        with open(self.data_root + data_source + '/' + 'val_indexes.list', 'r') as fp:
            val_indexes = fp.readlines()

        with open(self.data_root + data_source + '/' + 'test_contents.list', 'r') as fp:
            test_contents = fp.readlines()

        with open(self.data_root + data_source + '/' + 'test_labels.list', 'r') as fp:
            test_labels = fp.readlines()

        with open(self.data_root + data_source + '/' + 'test_indexes.list', 'r') as fp:
            test_indexes = fp.readlines()

        return all_contents, all_labels, \
               train_contents, train_labels, train_indexes, \
               val_contents, val_labels, val_indexes, \
               test_contents, test_labels, test_indexes

    def extract_count_features(self, train_contents, path=None, min_df=1):
        # 词频
        vectorizer = CountVectorizer(min_df=min_df)
        vectorizer.fit(train_contents)
        # print(vectorizer.vocabulary_)
        if not path:
            dump(vectorizer, './checkpoints/{}/tf_vectorizer'.format(self.data_source))
        else:
            dump(vectorizer, path)
        return vectorizer

    def extract_tfidf_features(self, train_contents, path=None):
        # TF-IDF
        vectorizer = TfidfVectorizer()
        vectorizer.fit(train_contents)
        if not path:
            dump(vectorizer, './checkpoints/{}/tfidf_vectorizer'.format(self.data_source))
        else:
            dump(vectorizer, path)
        return vectorizer

    def extract_tfidf_features_gensim(self, train_contents, path=None):
        # TF-IDF
        train_contents = [self.tokenizer(content) for content in train_contents]
        gensim_dict = Dictionary(train_contents)
        train_corpus = [gensim_dict.doc2bow(content) for content in train_contents]
        vectorizer = TfidfModel(train_corpus)
        if not path:
            dump(vectorizer, './checkpoints/{}/tfidf_vectorizer_gensim'.format(self.data_source))
            dump(gensim_dict, './checkpoints/{}/tfidf_vectorizer_gensim_dict'.format(self.data_source))
        else:
            dump(vectorizer, path)
            dump(gensim_dict, path + '_dict')
        return gensim_dict, vectorizer

    def extract_lda_features(self, train_contents, num_topics=20, path=None):
        # lda
        train_contents = [self.tokenizer(content) for content in train_contents]
        lda_dict = Dictionary(train_contents)
        train_corpus = [lda_dict.doc2bow(content) for content in train_contents]
        lda = LdaModel(train_corpus, num_topics=num_topics)

        if not path:
            dump(lda, './checkpoints/{}/lda'.format(self.data_source))
            dump(lda_dict, './checkpoints/{}/lda_dict'.format(self.data_source))
        else:
            dump(lda, path)
            dump(lda_dict, path + '_dict')
        return lda_dict, lda

    def get_word2vec(self, train_contents, embed_dim, path=None, sg=0):
        print('using word2vec...')
        train_contents = [self.tokenizer(content) for content in train_contents]
        model = Word2Vec(sentences=train_contents, vector_size=embed_dim, window=5, min_count=1, workers=4, sg=sg)
        if not path:
            model.wv.save_word2vec_format('./checkpoints/{}/word2vec'.format(self.data_source))
            model.save('./checkpoints/{}/word2vec_model'.format(self.data_source))
        else:
            model.wv.save_word2vec_format(path)
            model.save(path + '_model')
        return model

    def get_X_Y(self, vectorizer, contents, labels):
        X = vectorizer.transform(contents)
        Y = list(map(lambda x: int(x.strip()), labels))
        vocabs = vectorizer.vocabulary_
        result_dict = {}
        pos = np.array(list(filter(lambda x: int(x[1]) == 1, list(zip(X.todense(), labels)))))[:, 0]
        neg = np.array(list(filter(lambda x: int(x[1]) == 0, list(zip(X.todense(), labels)))))[:, 0]
        print(pos.shape)
        print(neg.shape)
        result_dict['pos'] = np.array(np.sum(pos, axis=0, keepdims=False)).flatten()
        result_dict['neg'] = np.array(np.sum(neg, axis=0, keepdims=False)).flatten()
        # print(result_dict['pos'].shape)
        # print(result_dict['neg'].shape)
        label_count = pd.DataFrame(index=vocabs, data=result_dict)
        print(label_count)
        return X, np.array(Y)

    def get_gensim_X_Y(self, gensim_dict, tfidf_model, contents, labels):
        num_vocab = len(gensim_dict.keys())
        X = [np.array(tfidf_model[gensim_dict.doc2bow(self.tokenizer(content))])
             for content in contents]
        X = [
            coo_matrix(
                (line[:, 1], (np.array([0] * line.shape[0]), np.array(line[:, 0], dtype=int))),
                shape=(1, num_vocab), dtype=float
            ) if len(line.shape) >= 2 else
            coo_matrix(
                ([0], (np.array([0]), np.array([0]))),
                shape=(1, num_vocab), dtype=float
            )
            for line in X
        ]
        # X = [[prob for (topic, prob) in content] for content in X]
        Y = list(map(lambda x: int(x.strip()), labels))
        return X, np.array(Y)

    def get_lda_X_Y(self, lda_dict, lda_model, contents, labels):
        X = [lda_model.get_document_topics(lda_dict.doc2bow(self.tokenizer(content)), minimum_probability=0)
             for content in contents]
        X = [[prob for (topic, prob) in content] for content in X]
        Y = list(map(lambda x: int(x.strip()), labels))
        return X, np.array(Y)

    @staticmethod
    def yield_tokens(data_iter):
        for content in data_iter:
            yield tokenizer(content)

    def build_vocab(self, train_contents, word2vec_path=None, load_vocab=False, BERT_tokenizer_path=None,
                    save_path=None):
        if word2vec_path:
            vec = Vectors(word2vec_path)
            self.vocab = vocab(vec.stoi, min_freq=0)  # 这里的转换把index当成了freq，为保证一一对应设置为0，实际上不影响后续操作
            self.vocab.append_token(UNK)
            self.vocab.append_token(PAD)
            unk_vec = torch.mean(vec.vectors, dim=0).unsqueeze(0)
            pad_vec = torch.zeros(vec.vectors.shape[1]).unsqueeze(0)
            self.vectors = torch.cat([vec.vectors, unk_vec, pad_vec])
        elif BERT_tokenizer_path:
            self.tokenizer = BertTokenizer.from_pretrained(BERT_tokenizer_path)
            self.vocab = vocab(self.tokenizer.vocab, min_freq=0)
        else:
            if load_vocab:
                self.vocab = torch.load(self.data_root + self.data_source + '/' + 'vocab.pth')
            else:
                self.vocab = build_vocab_from_iterator(self.yield_tokens(train_contents), specials=[UNK, PAD])
        self.vocab.set_default_index(self.vocab[UNK])
        # self.text_pipeline = lambda x: self.vocab(tokenizer(x.strip()))
        # self.label_pipeline = lambda x: int(x.strip())
        if not save_path:
            torch.save(self.vocab, self.data_root + self.data_source + '/' + 'vocab.pth')
        else:
            torch.save(self.vocab, save_path)
        return self.vocab

    def get_dataloader(self, batch_size, extract_method='raw_text', cut=False, using_bert=False):
        all_contents, all_labels, \
        train_contents, train_labels, train_indexes, \
        val_contents, val_labels, val_indexes, \
        test_contents, test_labels, test_indexes = self.load_data()
        print(len(train_contents))

        self.label_pipeline = lambda x: int(x.strip())
        if cut:
            # print(self.pad_size)
            self.text_pipeline = lambda x: self.get_dealt_text(x, self.pad_size, using_bert)
        else:
            self.text_pipeline = lambda x: self.get_dealt_text(x, using_bert)

        train_dataloader = DataLoader(list(zip(train_contents, train_labels, train_indexes)), batch_size=batch_size,
                                      shuffle=True, collate_fn=self.collate_batch)
        val_dataloader = DataLoader(list(zip(val_contents, val_labels, val_indexes)), batch_size=batch_size,
                                    shuffle=True, collate_fn=self.collate_batch)
        test_dataloader = DataLoader(list(zip(test_contents, test_labels, test_indexes)), batch_size=batch_size,
                                     shuffle=True, collate_fn=self.collate_batch)

        self.dataloaders = [train_dataloader, val_dataloader, test_dataloader]

        return train_dataloader, val_dataloader, test_dataloader

    def get_dealt_text(self, text, pad_size=None, using_BERT=True):
        mask = []
        if using_BERT:
            tokens = self.tokenizer.tokenize(text.strip())
            tokens = [CLS] + tokens + [SEP]
            seq_len = len(tokens)
        else:
            tokens = self.tokenizer(text.strip())
            seq_len = len(tokens)

        if pad_size:
            if len(tokens) < pad_size:
                tokens.extend([PAD] * (pad_size - len(tokens)))
                # mask = [1] * len(tokens) + [0] * (pad_size - len(tokens))
                mask = [1] * seq_len + [0] * (pad_size - seq_len)
            else:
                tokens = tokens[:pad_size]
                if using_BERT:
                    tokens[-1] = SEP
                seq_len = pad_size
                mask = [1] * pad_size
        return torch.tensor(self.vocab(tokens), dtype=torch.int64), seq_len, mask

    def collate_batch(self, batch):
        label_list, content_list = [], []
        length_list = []
        mask_list = []
        index_list = []
        PAD_IDX = self.vocab[PAD]
        for (_content, _label, _index) in batch:
            processed_content, seq_len, mask = self.text_pipeline(_content)
            # print(_label)
            label_list.append(self.label_pipeline(_label))
            content_list.append(processed_content)
            length_list.append(seq_len)
            mask_list.append(mask)
            index_list.append(int(_index.strip()))
        # content_list = torch.cat(content_list)
        content_batch = pad_sequence(content_list, padding_value=PAD_IDX).to(device)
        label_list = torch.tensor(label_list, dtype=torch.int64)
        length_list = torch.tensor(length_list, dtype=torch.int64)
        mask_list = torch.tensor(mask_list, dtype=torch.int8)
        index_list = torch.tensor(index_list, dtype=torch.int64)
        # print(len(label_list))
        return content_batch.to(device), label_list.to(device), length_list.to(device), mask_list.to(
            device), index_list.to(device)

    def get_full_dealt_data(self, config, flatten=False, data_name='save_data', max_count=None, seed=123):
        if not self.data_source.endswith('full'):
            print('What are you doing?')
        # text_dict = load_txt_vocab(self.data_cat_path + 'text_dict')
        # authors_dict = load_txt_vocab(self.data_cat_path + 'authors_dict')
        label_dict = json.load(open(self.data_cat_path + 'label_map.json', 'r'))
        phases = ['train', 'val', 'test']
        parts = ['title', 'intro', 'related', 'methods', 'conclusion', 'abstract']
        doc_parts = ['intro', 'related', 'methods', 'conclusion', 'abstract']
        full_len = config['full_len']
        if flatten:
            print('ops')
        else:
            self.vocab = torch.load(
                self.data_root + self.data_source + '/' + 'vocab.pth')
            self.authors_dict = torch.load(
                self.data_root + self.data_source + '/' + 'authors_vocab.pth')
            phase_dict = {}
            for phase in phases:
                part_dict = {}
                part_temp_list = []
                # final_tokens_list = []
                # final_len_list = []
                print(label_dict)
                final_label_list = json.load(open(self.data_cat_path + '{}_label_{}'.format(phase, seed), 'r'))
                final_label_list = [int(label_dict.get(str(label))) for label in final_label_list]
                final_index_list = json.load(open(self.data_cat_path + '{}_index_{}'.format(phase, seed), 'r'))
                final_index_list = [int(index) for index in final_index_list]
                if max_count:
                    final_label_list = final_label_list[:max_count]
                    final_index_list = final_index_list[:max_count]
                # final_label_list = np.array([np.array([int(label_dict.get(label))]) for label in final_label_list])
                for part in doc_parts:
                    with open(self.data_cat_path + '{}_{}_{}'.format(phase, part, seed), 'r') as fr:
                        temp_list = json.load(fr)
                    if max_count:
                        temp_list = temp_list[:max_count]
                    result_list = []
                    for line in tqdm(temp_list):
                        doc_array = np.zeros((full_len[part][0], full_len[part][1]), dtype=np.int32)
                        i = 0
                        for sent in line:
                            if i < full_len[part][0]:
                                tokens = self.vocab(self.tokenizer(sent.encode('utf-8', 'replace').decode('utf-8')))
                                if len(tokens) >= 5:
                                    for j in range(len(tokens)):
                                        if j < full_len[part][1]:
                                            doc_array[i, j] = tokens[j]
                                    i += 1
                                else:
                                    continue
                            else:
                                continue
                        result_list.append(doc_array)
                    print('dealt with {}'.format(part))
                    part_temp_list.append(result_list)
                for part in [part for part in parts if part not in doc_parts]:
                    with open(self.data_cat_path + '{}_{}_{}'.format(phase, part, seed), 'r') as fr:
                        temp_list = json.load(fr)
                    if max_count:
                        temp_list = temp_list[:max_count]
                    result_list = []
                    for line in temp_list:
                        doc_array = np.zeros(full_len[part][0] * full_len[part][1], dtype=np.int32)
                        tokens = self.vocab(self.tokenizer(line.encode('utf-8', 'replace').decode('utf-8')))
                        for i in range(len(tokens)):
                            if i < len(doc_array):
                                doc_array[i] = tokens[i]
                        result_list.append(doc_array)
                    print('dealt with {}'.format(part))
                    part_temp_list.append(result_list)
                with open(self.data_cat_path + '{}_{}_{}'.format(phase, 'authors', seed), 'r') as fr:
                    temp_list = json.load(fr)
                    result_list = []
                    for line in temp_list:
                        authors_array = np.zeros(full_len['authors'][0], dtype=np.int32)
                        authors = self.authors_dict(line.split(','))
                        for i in range(len(authors)):
                            if i < full_len['authors'][0]:
                                authors_array[i] = authors[i]
                        result_list.append(authors_array)
                    part_temp_list.append(result_list)
                # final_tokens_list = list(zip(part_temp_list[0], part_temp_list[1], part_temp_list[2], part_temp_list[3],
                #                              part_temp_list[4], part_temp_list[5], part_temp_list[6]))
                final_tokens_list = part_temp_list
                final_len_list = [0] * len(final_label_list)
                final_mask_list = [[0]] * len(final_label_list)
                # part_dict['contents'] = final_tokens_list
                # part_dict['lens'] = np.array(final_len_list, dtype=np.int32)
                # part_dict['labels'] = np.array(final_label_list, dtype=np.int8)
                # phase_dict[phase] = part_dict
                part_dataset = dataset(final_tokens_list, final_label_list, final_len_list, final_mask_list,
                                       final_index_list)
                phase_dict[phase] = part_dataset

        torch.save(phase_dict, self.data_cat_path + data_name)

    def get_saved_dataloader(self, base_config):
        data_dict = torch.load(self.data_cat_path + base_config['saved_data'])
        self.content_num = len(base_config['full_len'])
        self.vocab = torch.load(self.data_cat_path + 'vocab.pth')
        self.authors_dict = torch.load(self.data_cat_path + 'authors_vocab.pth')
        self.text_num = base_config['text_numbers']
        self.ablation_list = base_config['ablation_list']

        train_dataloader = DataLoader(dataset=data_dict['train'], batch_size=base_config['batch_size'],
                                      shuffle=True, num_workers=1, collate_fn=self.collate_batch_saved)
        val_dataloader = DataLoader(dataset=data_dict['val'], batch_size=base_config['batch_size'],
                                    shuffle=True, num_workers=1, collate_fn=self.collate_batch_saved)
        test_dataloader = DataLoader(dataset=data_dict['test'], batch_size=base_config['batch_size'],
                                     shuffle=True, num_workers=1, collate_fn=self.collate_batch_saved)
        self.dataloaders = [train_dataloader, val_dataloader, test_dataloader]

        return train_dataloader, val_dataloader, test_dataloader

    def collate_batch_saved(self, batch):
        inputs = zip(*batch)
        inputs_list = [samples for samples in inputs]
        result_lists = []
        contents = zip(*inputs_list[0])
        dealt_contents = []
        content_index = 0
        for content in contents:
            if content_index in self.ablation_list:
                UNK_TOKEN = self.vocab[UNK] if content_index < 6 else self.authors_dict[UNK]
                dealt_contents.append(torch.stack(
                    [torch.full(content[0].shape, UNK_TOKEN, dtype=torch.int32)] * len(content)
                ))
            else:
                dealt_contents.append(torch.stack(content))
            content_index += 1
        result_lists.append(dealt_contents)
        for i in range(1, len(inputs_list)):
            result_lists.append(torch.stack(inputs_list[i]))
        return result_lists

    def get_han_dataloader(self, base_config):
        data_dict = torch.load(self.data_cat_path + base_config['saved_data'])
        self.content_num = len(base_config['full_len'])
        self.vocab = torch.load(self.data_cat_path + 'vocab.pth')

        self.special_tokens = ['<TITLE>', '</TITLE>', '<ABSTRACT>', '</ABSTRACT>', '<BODY_TEXT>', '</BODY_TEXT>']
        for token in self.special_tokens:
            self.vocab.append_token(token)
        torch.save(self.vocab, self.data_cat_path + 'vocab_han.pth')

        train_dataloader = DataLoader(dataset=data_dict['train'], batch_size=base_config['batch_size'],
                                      shuffle=True, num_workers=0, collate_fn=self.collate_batch_han)
        val_dataloader = DataLoader(dataset=data_dict['val'], batch_size=base_config['batch_size'],
                                    shuffle=True, num_workers=0, collate_fn=self.collate_batch_han)
        test_dataloader = DataLoader(dataset=data_dict['test'], batch_size=base_config['batch_size'],
                                     shuffle=True, num_workers=0, collate_fn=self.collate_batch_han)
        self.dataloaders = [train_dataloader, val_dataloader, test_dataloader]

        return train_dataloader, val_dataloader, test_dataloader

    def collate_batch_han(self, batch):
        title_start, title_end, abs_start, abs_end, bt_start, bt_end = self.special_tokens
        inputs_list = list(zip(*batch))
        result_lists = []
        contents = inputs_list[0]
        # print(len(contents))
        # print(len(contents[0]))
        dealt_contents = []
        dealt_ls = []
        dealt_lr = []
        for content in contents:
            intro, related, methods, conclu, abstract, title, _ = content
            title = F.pad(title.unsqueeze(dim=0), (0, 20))
            text = [title, abstract, intro, related, methods, conclu]
            ls = [(part > 0).sum(dim=1) for part in text]
            lr = [(part > 0).sum(dim=0).item() for part in ls]
            # print(lr)
            # print(sum(lr))
            ls = [(abstract.shape[1] - torch.argmax((torch.fliplr(part) > 0).int(), dim=1))[:lr[idx]] for (idx, part) in
                  enumerate(text)]
            # print(ls)

            text = [part[:lr[idx]] for (idx, part) in enumerate(text)]

            title_text = text[0]
            title_ls = ls[0]
            title_text, title_ls = get_appended_tokens(title_text, title_ls, self.vocab, [title_start, title_end])

            abstract_text = text[1]
            abstract_ls = ls[1]
            abstract_text, abstract_ls = get_appended_tokens(abstract_text, abstract_ls, self.vocab, [abs_start, abs_end])

            body_text = torch.cat(text[2:], dim=0)
            body_ls = torch.cat(ls[2:], dim=0)
            body_text, body_ls = get_appended_tokens(body_text, body_ls, self.vocab, [bt_start, bt_end])

            final_text = torch.cat((title_text, abstract_text, body_text), dim=0)
            final_ls = torch.cat((title_ls, abstract_ls, body_ls))
            final_lr = torch.tensor(final_text.shape[0], dtype=torch.int32)

            dealt_contents.append(final_text)
            dealt_ls.append(final_ls)
            dealt_lr.append(final_lr)

        dealt_contents = pad_sequence(dealt_contents, batch_first=True)
        dealt_ls = pad_sequence(dealt_ls, batch_first=True, padding_value=1)
        dealt_lr = torch.tensor(dealt_lr, dtype=torch.int32)

        result_lists.append(dealt_contents)
        for i in range(1, len(inputs_list)):
            result_lists.append(torch.stack(inputs_list[i]))
        result_lists[2] = (dealt_ls, dealt_lr)
        # print(len(result_lists))
        return result_lists

    def get_chunk_dataloader(self, base_config):
        data_dict = torch.load(self.data_cat_path + base_config['chunk_data'])
        # print(data_dict['test'])
        train_dataloader = DataLoader(dataset=list(zip(data_dict['train']['text'], data_dict['train']['labels'],
                                                      data_dict['train']['lengths'], data_dict['train']['masks'],
                                                      data_dict['train']['indexes'])),
                                     batch_size=base_config['batch_size'],
                                     shuffle=True, num_workers=1, collate_fn=self.collate_batch_chunk)
        val_dataloader = DataLoader(dataset=list(zip(data_dict['val']['text'], data_dict['val']['labels'],
                                                      data_dict['val']['lengths'], data_dict['val']['masks'],
                                                      data_dict['val']['indexes'])),
                                     batch_size=base_config['batch_size'],
                                     shuffle=True, num_workers=1, collate_fn=self.collate_batch_chunk)
        test_dataloader = DataLoader(dataset=list(zip(data_dict['test']['text'], data_dict['test']['labels'],
                                                      data_dict['test']['lengths'], data_dict['test']['masks'],
                                                      data_dict['test']['indexes'])),
                                     batch_size=base_config['batch_size'],
                                     shuffle=True, num_workers=1, collate_fn=self.collate_batch_chunk)
        self.dataloaders = [train_dataloader, val_dataloader, test_dataloader]
        #
        return train_dataloader, val_dataloader, test_dataloader

    def collate_batch_chunk(self, batch):
        # inputs = zip(*batch)
        # inputs_list = [samples for samples in inputs]
        inputs_list = list(zip(*batch))
        # print(len(inputs_list))
        # print(inputs_list[0])
        contents = pad_sequence(inputs_list[0], batch_first=True)
        result_lists = []
        # dealt_contents = []
        result_lists.append(contents)
        for i in range(1, len(inputs_list)):
            if type(inputs_list[i][0]) == str:
                inputs_list[i] = [int(i.strip()) for i in inputs_list[i]]
            result_lists.append(torch.tensor(inputs_list[i], dtype=torch.int32))

        return result_lists


class dataset(Dataset):

    def __init__(self, text_data, label_data, length_data, mask_data, index_data):
        self.text_data = text_data
        self.label_data = label_data
        self.length_data = length_data
        self.mask_data = mask_data
        self.index_data = index_data

    def __getitem__(self, index):
        return [torch.from_numpy(x[index]) for x in self.text_data], \
               torch.tensor(self.label_data[index], dtype=torch.int8), \
               torch.tensor(0, dtype=torch.int8), torch.tensor([0], dtype=torch.int8), \
               torch.tensor(self.index_data[index], dtype=torch.int32)

    def __len__(self):
        return len(self.label_data)


def get_appended_tokens(text, ls, vocab, spec_tokens):
    start, end = spec_tokens
    start_idx = vocab[start]
    end_idx = vocab[end]
    text = F.pad(text, (1, 1))

    text[:, 0] = torch.zeros(text.shape[0], dtype=torch.int32) + start_idx
    seq_list = list(range(text.shape[0]))
    text[seq_list, ls + 1] = torch.zeros(text.shape[0], dtype=torch.int32) + end_idx
    ls = ls + 2
    # print(text)
    # print(ls)
    return text, ls

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')
    parser.add_argument('--seed', default=333, help='the seed.')
    parser.add_argument('--data_source', default='PeerRead_full', help='the data source.')
    parser.add_argument('--in_path', default=None, help='the input.')
    parser.add_argument('--out_path', default=None, help='the output.')

    args = parser.parse_args()

    dataProcessor = DataProcessor(data_source=args.data_source)
    # dataProcessor.split_data()
    # dataProcessor.load_data()

    if args.phase == 'test':
        print('This is a test process.')
        config = json.load(open('./configs/{}.json'.format('PeerRead_full')))['default']
        # dataloader = dataProcessor.get_chunk_dataloader(config)
        dataloader = dataProcessor.get_han_dataloader(config)[0]
        for idx, (content, label, lens, mask, index) in enumerate(dataloader):
            print(content.shape)
            print(lens)
            print(label)

    elif args.phase == 'extract_data':
        dataProcessor.extract_data()
    elif args.phase == 'label_map':
        dataProcessor.label_map()
    elif args.phase == 'split_data':
        dataProcessor.split_data()
    elif args.phase == 'get_dataloader':
        train_dataloader, val_dataloader, test_dataloader = dataProcessor.get_dataloader(32, cut=True)
        # for idx, (content, label, lens) in enumerate(train_dataloader):
        #     print(len(content))
        #     print(lens)
    elif args.phase == 'get_aapr':
        DATA_SOURCE = 'AAPR_full'
        config = json.load(open('./configs/{}.json'.format(DATA_SOURCE)))['default']
        seed = int(args.seed)
        print('data_seed', seed)
        in_path = args.in_path
        out_path = args.out_path
        make_data(in_path, out_path, seed, config['rate'])
        dataProcessor = DataProcessor(data_source=DATA_SOURCE)
        dataProcessor.get_full_dealt_data(config, False, 'save_data')
    elif args.phase == 'get_peerread':
        DATA_SOURCE = 'PeerRead_full'
        config = json.load(open('./configs/{}.json'.format(DATA_SOURCE)))['default']
        seed = int(args.seed)
        print('data_seed', seed)
        in_path = args.in_path
        out_path = args.out_path
        make_pr_data(in_path, out_path, seed, config['rate'])
        dataProcessor = DataProcessor(data_source=DATA_SOURCE)
        dataProcessor.get_full_dealt_data(config, False, 'save_data')
    elif args.phase == 'get_chunk':
        config = json.load(open('./configs/{}.json'.format(args.data_source)))['default']
        seed = int(args.seed)
        print('data_seed', seed)
        in_path = './data/{}/'.format(args.data_source)
        out_path = './data/{}/'.format(args.data_source)
        make_chunk_data(in_path, out_path, seed, './bert/base_bert/')

    else:
        print('error! No such method!')
    end_time = datetime.datetime.now()
    print('{} takes {} seconds'.format(args.phase, (end_time - start_time).seconds))

    print('Done data_processor!')
