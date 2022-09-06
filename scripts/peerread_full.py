import sys
import os

sys.path.append(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0])

from scripts.aapr_full import *

# print(parts)
datasets = ['arxiv.cs.ai_2007-2017', 'arxiv.cs.cl_2007-2017', 'arxiv.cs.lg_2007-2017']
# data_root = './data/PeerRead_full/'

# sample = json.load(open(data_root + datasets[0] + '_full.json'))
# print(sample['test'][0]['title'])


def split_pr_data(in_path, out_path, seed=333, rate=0.8):
    result_list = []
    for dataset in datasets:
        result_list.append(json.load(open(in_path + '{}_full.json'.format(dataset), 'r')))
    phases = ['train', 'dev', 'test']
    result_dict = {phase: {} for phase in phases}
    count = 0
    for phase in phases:
        # phase_name = 'val' if phase == 'dev' else phase
        phase_result = []
        for i in range(len(datasets)):
            phase_result.extend(result_list[i][phase])
        df = pd.DataFrame(phase_result)
        print(df.keys())
        print('original data length', len(df))
        df = df[df['title'].notna()]
        df = df[df['abstract'].notna()]
        df['abstract_text'] = df['abstract'].apply(lambda x: ' '.join(x))
        df = df[df['abstract_text'].apply(lambda x: len(tokenizer(x.strip())) > 20)]
        print(df[['intro', 'related', 'methods', 'conclusion', 'authors']].describe())
        print(df[df['methods'].apply(lambda x: len(x)==0)][['intro', 'related', 'methods', 'conclusion', 'authors']].head())
        print('selected data length', len(df))
        contents = [df[part].to_list() for part in parts]
        result_dict[phase]['contents'] = list(zip(*contents))
        print(len(result_dict[phase]['contents']))
        print(len(result_dict[phase]['contents'][0]))
        result_dict[phase]['labels'] = df['label'].astype(int).to_list()
        result_dict[phase]['indexes'] = list(range(count, count + len(df)))
        count += len(df)

    train_contents, val_contents, test_contents = result_dict['train']['contents'], \
                                                  result_dict['dev']['contents'], \
                                                  result_dict['test']['contents']
    train_labels, val_labels, test_labels = result_dict['train']['labels'], \
                                            result_dict['dev']['labels'], \
                                            result_dict['test']['labels']
    train_indexes, val_indexes, test_indexes = result_dict['train']['indexes'], \
                                               result_dict['dev']['indexes'], \
                                               result_dict['test']['indexes']

    contents = [train_contents, val_contents, test_contents]
    labels = [train_labels, val_labels, test_labels]
    indexes = [train_indexes, val_indexes, test_indexes]

    phases = ['train', 'val', 'test']
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


def make_pr_data(in_path, out_path, seed, rate):
    split_pr_data(in_path, out_path, seed=seed, rate=rate)
    get_vocab(out_path, seed)


if __name__ == '__main__':
    # split_pr_data('./data/PeerRead_full/', './data/PeerRead_full')
    in_path = './data/PeerRead_full/'
    out_path = './data/PeerRead_full'
    result_list = []
    for dataset in datasets:
        result_list.append(json.load(open(in_path + '{}_full.json'.format(dataset), 'r')))
    phases = ['train', 'dev', 'test']
    result_dict = {phase: {} for phase in phases}
    count = 0
    df_all = pd.DataFrame(columns=['intro', 'related', 'methods', 'conclusion', 'authors'])
    for phase in phases:
        # phase_name = 'val' if phase == 'dev' else phase
        phase_result = []
        for i in range(len(datasets)):
            phase_result.extend(result_list[i][phase])
        df = pd.DataFrame(phase_result)
        print(df.keys())
        print('original data length', len(df))
        df = df[df['title'].notna()]
        df = df[df['abstract'].notna()]
        df['abstract_text'] = df['abstract'].apply(lambda x: ' '.join(x))
        df = df[df['abstract_text'].apply(lambda x: len(tokenizer(x.strip())) > 20)]
        print(df[['intro', 'related', 'methods', 'conclusion', 'authors']].describe())
        df_all = df_all.append(df[['intro', 'related', 'methods', 'conclusion', 'authors']])
        print(df[df['methods'].apply(lambda x: len(x)==0)][['intro', 'related', 'methods', 'conclusion', 'authors']].head())
        print('selected data length', len(df))
        contents = [df[part].to_list() for part in parts]
        result_dict[phase]['contents'] = list(zip(*contents))
        print(len(result_dict[phase]['contents']))
        print(len(result_dict[phase]['contents'][0]))
        result_dict[phase]['labels'] = df['label'].astype(int).to_list()
        result_dict[phase]['indexes'] = list(range(count, count + len(df)))
        count += len(df)

    print(df_all.head())
    print(df_all.describe())
    print(df_all[df_all.apply(lambda x: (len(x['intro']) > 0) & (len(x['related']) > 0) & (len(x['conclusion']) > 0) &
                                        (len(x['methods']) > 0) & (len(x['authors']) > 0), axis=1)])
