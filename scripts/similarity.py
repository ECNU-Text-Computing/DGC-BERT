import argparse
import json
import sys
import os

import numpy as np
from gensim.models import Word2Vec

sys.path.append(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0])
from data_processor import DataProcessor
import pandas as pd
from torch import nn
from torchtext.data import get_tokenizer
import torch
from scipy import stats
import joblib

tokenizer = get_tokenizer('basic_english')
parts = ['abstract', 'title', 'intro', 'related', 'conclusion', 'methods', 'title+abstract']


# data = json.load(open('../data/AAPR/dealt_data', 'r'))
# selected_data = {}
# for key in data.keys():
#     selected_data[key] = data[key][:1000]
# json.dump(selected_data, open('../data/AAPR/selected_data', 'w+'))


def get_cleaned_data(input_file, output_path):
    # data = json.load(open(input_file, 'r'))
    # print(data.keys())
    all_data = json.load(open(input_file, 'r'))
    print(all_data.keys())
    data = [{key: all_data[key][i] for key in all_data.keys()} for i in range(len(all_data['abstract']))]
    print('original data length', len(data))
    # 筛选摘要内容长度超过20的有效文本作为数据，和只用摘要部分的数据对齐
    selected_data = list(filter(lambda x: len(tokenizer('. '.join(x['abstract']).strip())) > 20, data))
    print('selected data length', len(selected_data))
    data_keys = all_data.keys()
    all_data = {key: [] for key in data_keys}
    for paper in selected_data:
        for key in data_keys:
            all_data[key].append(paper[key])
    all_contents = []
    for key in parts:
        if (key != 'title') & (key != 'title+abtract'):
            # 进一步筛选每句话超过5个词的才算有效句子
            all_data[key] = [[sent for sent in line if len(tokenizer(sent)) >= 5] for line in all_data[key]]
            all_data[key] = ['. '.join(line) for line in all_data[key]]
        all_contents.extend(all_data[key])
    json.dump(all_data, open(output_path + 'cleaned_data', 'w+'))
    json.dump(all_contents, open(output_path + 'cleaned_all_contents', 'w+'))


def get_cos_similarity(data, transformed_data, name, sparse=False):
    result_list = []
    for doc_id in range(len(data['abstract'])):
        temp_result = {}
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                src = parts[i]
                dst = parts[j]
                # 注意这里不会过滤作为两个部分之和中一个为0的情况，所以需要手动过滤！！！
                if (len(data[src][doc_id]) > 0) & (len(data[dst][doc_id]) > 0):
                    if sparse:
                        src_vec = sparse_trans(transformed_data[src][doc_id])
                        dst_vec = sparse_trans(transformed_data[dst][doc_id])
                    else:
                        src_vec = torch.Tensor(transformed_data[src][doc_id])
                        dst_vec = torch.Tensor(transformed_data[dst][doc_id])
                    temp_result['{}_{}'.format(src, dst)] = torch.cosine_similarity(src_vec, dst_vec).item()
                else:
                    temp_result['{}_{}'.format(src, dst)] = None
        result_list.append(temp_result)
    pd.DataFrame(result_list).to_csv('{}.csv'.format(name))


def get_kl_div(data, transformed_data, name, sparse=False):
    # 因为是有向的，所以计算也是有向的
    result_list = []
    for doc_id in range(len(data['abstract'])):
        temp_result = {}
        for i in range(len(parts)):
            for j in range(len(parts)):
                if i != j:
                    src = parts[i]
                    dst = parts[j]
                    if (len(data[src][doc_id]) > 0) & (len(data[dst][doc_id]) > 0):
                        # if sparse:
                        #     src_vec = transformed_data[src][doc_id].todense()
                        #     dst_vec = transformed_data[dst][doc_id].todense()
                        # else:
                        #     src_vec = np.array(transformed_data[src][doc_id])
                        #     dst_vec = np.array(transformed_data[dst][doc_id])
                        src_vec = torch.from_numpy(transformed_data[src][doc_id].todense())
                        dst_vec = torch.from_numpy(transformed_data[dst][doc_id].todense())
                        # temp_result['{}_{}'.format(src, dst)] = stats.entropy(src_vec,
                        #                                                       dst_vec)
                        temp_result['{}_{}'.format(src, dst)] = nn.KLDivLoss(reduction='sum') \
                            (torch.log_softmax(dst_vec, dim=-1), torch.softmax(src_vec, dim=-1)).item()
                    else:
                        temp_result['{}_{}'.format(src, dst)] = None
        result_list.append(temp_result)
    pd.DataFrame(result_list).to_csv('{}.csv'.format(name))


def get_wmd_similarity(data, transformed_data, name, model):
    result_list = []
    for doc_id in range(len(data['abstract'])):
        temp_result = {}
        for i in range(len(parts)):
            src = parts[i]
            # index = WmdSimilarity([transformed_data[src][doc_id]], model)
            for j in range(i + 1, len(parts)):
                dst = parts[j]
                # query = transformed_data[dst][doc_id]
                if (len(data[src][doc_id]) > 0) & (len(data[dst][doc_id]) > 0):
                    # temp_result['{}_{}'.format(src, dst)] = index[query][0][1]
                    temp_result['{}_{}'.format(src, dst)] = model.wv.wmdistance(transformed_data[src][doc_id],
                                                                                transformed_data[dst][doc_id])
                else:
                    temp_result['{}_{}'.format(src, dst)] = None
        result_list.append(temp_result)
    pd.DataFrame(result_list).to_csv('{}.csv'.format(name))


def sparse_trans(x):
    indices = torch.LongTensor(np.vstack((x.row, x.col)))
    values = torch.FloatTensor(x.data)
    shape = x.shape
    x_trans = torch.sparse.FloatTensor(indices, values, shape)
    return x_trans.to_dense()


# def get_tfidf(data, all_contents, dataProcessor):
#     transformed_data = {}
#     # tfidf
#     vectorizer = dataProcessor.extract_tfidf_features(all_contents, path='./checkpoints/test/tfidf_vectorizer')
#     for key in parts:
#         transformed_data[key], _ = dataProcessor.get_X_Y(vectorizer, data[key], ['0'] * len(data[key]))
#         transformed_data[key] = [line.todense() for line in transformed_data[key]]
#     get_similarity(data, transformed_data, './checkpoints/test/tfidf_sim')


def get_tfidf(data, all_contents, dataProcessor):
    transformed_data = {}
    # tfidf
    gensim_dict, vectorizer = dataProcessor.extract_tfidf_features_gensim(all_contents,
                                                                          path='./checkpoints/test/tfidf_vectorizer_gensim')
    for key in parts:
        transformed_data[key], _ = dataProcessor.get_gensim_X_Y(gensim_dict, vectorizer, data[key],
                                                                ['0'] * len(data[key]))
        # transformed_data[key] = [line.todense() for line in transformed_data[key]]
    get_cos_similarity(data, transformed_data, './checkpoints/test/tfidf_sim', sparse=True)


def get_kl(data, dataProcessor):
    transformed_data = {}
    # tfidf
    gensim_dict = joblib.load('./checkpoints/test/tfidf_vectorizer_gensim_dict')
    vectorizer = joblib.load('./checkpoints/test/tfidf_vectorizer_gensim')
    for key in parts:
        transformed_data[key], _ = dataProcessor.get_gensim_X_Y(gensim_dict, vectorizer, data[key],
                                                                ['0'] * len(data[key]))
        # transformed_data[key] = [line.todense() for line in transformed_data[key]]
    get_kl_div(data, transformed_data, './checkpoints/test/kl_div', sparse=True)


def get_lda(data, all_contents, dataProcessor, num_topics=20):
    transformed_data = {}
    # lda
    lda_dict, lda_model = dataProcessor.extract_lda_features(all_contents, num_topics=num_topics,
                                                             path='./checkpoints/test/lda')
    for key in parts:
        transformed_data[key], _ = dataProcessor.get_lda_X_Y(lda_dict, lda_model, data[key], ['0'] * len(data[key]))
        transformed_data[key] = [[line] for line in transformed_data[key]]
    get_cos_similarity(data, transformed_data, './checkpoints/test/lda_sim')


def get_glove(data, all_contents, dataProcessor):
    transformed_data = {}
    # glove
    dataProcessor.build_vocab(all_contents, word2vec_path='./checkpoints/test/glove',
                              save_path='./checkpoints/test/vocab_glove')
    embedding = nn.Embedding.from_pretrained(dataProcessor.vectors, freeze=False)
    for key in parts:
        transformed_data[key] = [torch.unsqueeze(torch.mean(
            embedding(torch.tensor(dataProcessor.vocab(tokenizer(line)), dtype=torch.int))
            , dim=0), dim=0)
            for line in data[key]]
    get_cos_similarity(data, transformed_data, './checkpoints/test/glove_sim')


def get_w2v(data, all_contents, dataProcessor):
    transformed_data = {}
    # w2v
    model = dataProcessor.get_word2vec(all_contents, 300, path='./checkpoints/test/w2v', sg=1)
    dataProcessor.build_vocab(all_contents, word2vec_path='./checkpoints/test/w2v',
                              save_path='./checkpoints/test/vocab_w2v')
    embedding = nn.Embedding.from_pretrained(dataProcessor.vectors, freeze=False)
    for key in parts:
        transformed_data[key] = [torch.unsqueeze(torch.mean(
            embedding(torch.tensor(dataProcessor.vocab(tokenizer(line)), dtype=torch.int))
            , dim=0), dim=0)
            for line in data[key]]
    get_cos_similarity(data, transformed_data, './checkpoints/test/w2v_sim')


def get_w2v_wdm(data):
    transformed_data = {}
    # w2v
    model = Word2Vec.load('./checkpoints/test/w2v_model')
    for key in parts:
        transformed_data[key] = [tokenizer(line) for line in data[key]]
    get_wmd_similarity(data, transformed_data, './checkpoints/test/w2v_wdm', model)


def get_word_frequency(data, all_contents, dataProcessor):
    # transformed_data = {}
    # 不同部分
    vectorizer = dataProcessor.extract_count_features(all_contents, path='./checkpoints/test/tf_vectorizer', min_df=5)
    # # for key in parts:
    # #     transformed_data[key], _ = dataProcessor.get_X_Y(vectorizer, data[key], data['label'])
    # # transformed_data['label'] = data['label']
    labels = data['label']
    # # joblib.dump(transformed_data, 'tf_dict')
    # # joblib.dump(np.array(data['label']), 'label_dict')
    count_dict = {}
    vocabs = vectorizer.vocabulary_
    print(len(vocabs))
    for key in parts:
        temp_matrix, _ = dataProcessor.get_X_Y(vectorizer, data[key], data['label'])
        count_dict[key] = np.sum(np.array(temp_matrix.todense()), axis=0)
    part_count = pd.DataFrame(index=vocabs, data=count_dict)
    part_count.to_csv('./checkpoints/test/part_count.csv')
    # 正负例
    abstracts, _ = dataProcessor.get_X_Y(vectorizer, data['title+abstract'], data['label'])
    result_dict = {}
    pos = np.array(list(filter(lambda x: int(x[1]) == 1, list(zip(abstracts.todense(), labels)))))[:, 0]
    neg = np.array(list(filter(lambda x: int(x[1]) == 0, list(zip(abstracts.todense(), labels)))))[:, 0]
    print(pos.shape)
    print(neg.shape)
    result_dict['pos'] = np.array(np.sum(pos, axis=0, keepdims=False)).flatten()
    result_dict['neg'] = np.array(np.sum(neg, axis=0, keepdims=False)).flatten()
    # print(result_dict['pos'].shape)
    # print(result_dict['neg'].shape)
    label_count = pd.DataFrame(index=vocabs, data=result_dict)
    label_count.to_csv('./checkpoints/test/label_count.csv')


def get_all_similarity(path, num_topics=10):
    data = json.load(open(path + 'cleaned_data', 'r'))
    data['title+abstract'] = [data['title'][i] + '. ' + data['abstract'][i] for i in range(len(data['abstract']))]
    all_contents = json.load(open(path + 'cleaned_all_contents', 'r'))
    dataProcessor = DataProcessor(data_source='test')
    get_tfidf(data, all_contents, dataProcessor)
    get_lda(data, all_contents, dataProcessor, num_topics=num_topics)
    get_glove(data, all_contents, dataProcessor)
    get_w2v(data, all_contents, dataProcessor)


def get_label_similarity(path, num_topics=50):
    data = json.load(open(path + 'cleaned_data', 'r'))
    dataProcessor = DataProcessor(data_source='test')
    all_abstracts = [data['title'][i] + '. ' + data['abstract'][i] for i in range(len(data['abstract']))]
    all_labels = data['label']
    pos = np.array(list(filter(lambda x: int(x[1]) == 1, list(zip(all_abstracts, all_labels)))))[:, 0].tolist()
    neg = np.array(list(filter(lambda x: int(x[1]) == 0, list(zip(all_abstracts, all_labels)))))[:, 0].tolist()
    print('pos', len(pos))
    print('neg', len(neg))
    result_dict = {}

    gensim_dict, vectorizer = dataProcessor.extract_tfidf_features_gensim(all_abstracts,
                                                                          path='./checkpoints/test/tfidf_ta_vectorizer_gensim')
    pos_vec, _ = dataProcessor.get_gensim_X_Y(gensim_dict, vectorizer, pos, ['0'] * len(pos))
    neg_vec, _ = dataProcessor.get_gensim_X_Y(gensim_dict, vectorizer, neg, ['0'] * len(neg))
    pos_vec = torch.from_numpy(np.mean(np.array([vec.todense() for vec in pos_vec]), axis=0, keepdims=False))
    neg_vec = torch.from_numpy(np.mean(np.array([vec.todense() for vec in neg_vec]), axis=0, keepdims=False))
    print(pos_vec.shape)
    result_dict['tfidf_sim'] = torch.cosine_similarity(pos_vec, neg_vec).item()
    print(result_dict['tfidf_sim'])
    # pos_vec = torch.softmax(torch.from_numpy(pos_vec), dim=-1).numpy()
    # neg_vec = torch.softmax(torch.from_numpy(neg_vec), dim=-1).numpy()
    # result_dict['tfidf_kl'] = (stats.entropy(pos_vec[0], neg_vec[0]) + stats.entropy(neg_vec[0], pos_vec[0]))/2
    result_dict['tfidf_kl'] = ((nn.KLDivLoss(reduction='sum')(torch.log_softmax(pos_vec, dim=-1),
                                                              torch.softmax(neg_vec, dim=-1)) +
                                nn.KLDivLoss(reduction='sum')(torch.log_softmax(neg_vec, dim=-1),
                                                              torch.softmax(pos_vec, dim=-1))) / 2).item()
    print(result_dict['tfidf_kl'])

    lda_dict, lda_model = dataProcessor.extract_lda_features(all_abstracts, num_topics=num_topics,
                                                             path='./checkpoints/test/ta_lda')
    pos_vec, _ = dataProcessor.get_lda_X_Y(lda_dict, lda_model, pos, ['0'] * len(pos))
    neg_vec, _ = dataProcessor.get_lda_X_Y(lda_dict, lda_model, neg, ['0'] * len(neg))
    pos_vec = np.mean(np.array([[vec] for vec in pos_vec]), axis=0, keepdims=False)
    neg_vec = np.mean(np.array([[vec] for vec in neg_vec]), axis=0, keepdims=False)
    print(pos_vec.shape)
    result_dict['lda_sim'] = torch.cosine_similarity(torch.from_numpy(pos_vec), torch.from_numpy(neg_vec)).item()
    print(result_dict['lda_sim'])

    dataProcessor.build_vocab(all_abstracts, word2vec_path='./checkpoints/test/glove',
                              save_path='./checkpoints/test/ta_vocab_glove')
    embedding = nn.Embedding.from_pretrained(dataProcessor.vectors, freeze=False)
    pos_vec = torch.cat([torch.unsqueeze(torch.mean(
        embedding(torch.tensor(dataProcessor.vocab(tokenizer(line)), dtype=torch.int))
        , dim=0), dim=0)
        for line in pos], dim=0).mean(dim=0).unsqueeze(dim=0)
    neg_vec = torch.cat([torch.unsqueeze(torch.mean(
        embedding(torch.tensor(dataProcessor.vocab(tokenizer(line)), dtype=torch.int))
        , dim=0), dim=0)
        for line in neg], dim=0).mean(dim=0).unsqueeze(dim=0)
    print(pos_vec.shape)
    result_dict['glove_sim'] = torch.cosine_similarity(pos_vec, neg_vec).item()
    print(result_dict['glove_sim'])

    model = dataProcessor.get_word2vec(all_abstracts, 300, path='./checkpoints/test/ta_w2v', sg=1)
    dataProcessor.build_vocab(all_abstracts, word2vec_path='./checkpoints/test/ta_w2v',
                              save_path='./checkpoints/test/vocab_ta_w2v')
    embedding = nn.Embedding.from_pretrained(dataProcessor.vectors, freeze=False)
    pos_vec = torch.cat([torch.unsqueeze(torch.mean(
        embedding(torch.tensor(dataProcessor.vocab(tokenizer(line)), dtype=torch.int))
        , dim=0), dim=0)
        for line in pos], dim=0).mean(dim=0).unsqueeze(dim=0)
    print(pos_vec.shape)
    neg_vec = torch.cat([torch.unsqueeze(torch.mean(
        embedding(torch.tensor(dataProcessor.vocab(tokenizer(line)), dtype=torch.int))
        , dim=0), dim=0)
        for line in neg], dim=0).mean(dim=0).unsqueeze(dim=0)
    result_dict['w2v_sim'] = torch.cosine_similarity(pos_vec, neg_vec).item()
    print(result_dict['w2v_sim'])

    df = pd.DataFrame(data=result_dict)
    df.to_csv('./checkpoints/test/label_sim.csv')


# bert = BertModel.from_pretrained('../bert/scibert/', output_hidden_states=True)
# tokenizer = BertTokenizer.from_pretrained('../bert/scibert/vocab.txt')
#
# tokens = tokenizer.tokenize('this paper deals with the average complexity of propositional proofs.')
# ids = tokenizer.convert_tokens_to_ids(tokens)
# print(tokens)
# print(bert(torch.tensor(ids).unsqueeze(dim=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')
    args = parser.parse_args()
    all_forms = {'tfidf', 'lda', 'glove', 'w2v', 'w2v_wmd', 'kl', 'frequency'}
    if args.phase == 'test':
        print('this is a test')
        # data = json.load(open('../data/AAPR/cleaned_data'))
        # print(data.keys())
        # sample = data['intro'][:10]
        # with open('sample.txt', 'w+') as fw:
        #     fw.write('\n'.join(sample))
        get_label_similarity('../data/AAPR/')
    elif args.phase == 'get_cleaned_data':
        get_cleaned_data('./data/AAPR/dealt_data', './data/AAPR/')
    elif args.phase == 'get_all_similarity':
        get_all_similarity('./data/AAPR/', num_topics=50)
    elif args.phase == 'label':
        get_label_similarity('./data/AAPR/', num_topics=50)
    elif args.phase in all_forms:
        path = './data/AAPR/'
        data = json.load(open(path + 'cleaned_data', 'r'))
        data['title+abstract'] = [data['title'][i] + '. ' + data['abstract'][i] for i in range(len(data['abstract']))]
        print(data['title+abstract'][0])
        all_contents = json.load(open(path + 'cleaned_all_contents', 'r'))
        dataProcessor = DataProcessor(data_source='test')
        if args.phase == 'tfidf':
            get_tfidf(data, all_contents, dataProcessor)
        elif args.phase == 'lda':
            get_lda(data, all_contents, dataProcessor, num_topics=20)
        elif args.phase == 'glove':
            get_glove(data, all_contents, dataProcessor)
        elif args.phase == 'w2v':
            get_w2v(data, all_contents, dataProcessor)
        elif args.phase == 'w2v_wmd':
            get_w2v_wdm(data)
        elif args.phase == 'kl':
            get_kl(data, dataProcessor)
        elif args.phase == 'frequency':
            get_word_frequency(data, all_contents, dataProcessor)
        print(args.phase + ' done!')
