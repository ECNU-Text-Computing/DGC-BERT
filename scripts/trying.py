import pandas as pd
from clean_latex import extract
import json

dataset = ['data1', 'data2', 'data3', 'data4']


def make_extracted_data(f_names):
    result_dict = {
        'abstract': [],
        'title': [],
        'authors': [],
        'label': [],
        'intro': [],
        'related': [],
        'conclusion': [],
        'methods': [],
        'old_abstract': []

    }
    for f_name in f_names:
        print('extracting ', f_name)
        data = json.load(open(f_name, 'r'))
        i = 0
        for key in data:
            line = data[key]
            print(i)
            i += 1
            get_decsion = lambda x: '0' if x in ['CoRR', 'No'] else '1'

            result_dict['title'].append(line['title'].lower().strip())
            result_dict['authors'].append(line['authors'].lower().strip())
            result_dict['label'].append(get_decsion(line['venue']))
            result_dict['old_abstract'].append(line['abstract'].lower().strip())

            extracted_data = extract(line['tex_data'])
            # print(len(extracted_data[0]))
            result_dict['abstract'].append([sent.strip() for sent in extracted_data[0]])
            result_dict['intro'].append([sent.strip() for sent in extracted_data[1]])
            result_dict['related'].append([sent.strip() for sent in extracted_data[2]])
            result_dict['methods'].append([sent.strip() for sent in extracted_data[3]])
            result_dict['conclusion'].append([sent.strip() for sent in extracted_data[4]])

    return result_dict


path = '../data/AAPR/'
# df = pd.read_json(path)
# df = df.T
# tex_data = df['tex_data'].to_list()
# dealt_data = make_extracted_data([path+fname for fname in dataset])
# json.dump(dealt_data, open('dealt_data', 'w'))
# data = json.load(open('sample', 'r'))
# print(len([line for line in data['related'] if ' '.join(line).strip() != '']))

all_data = json.load(open('dealt_data', 'r'))

