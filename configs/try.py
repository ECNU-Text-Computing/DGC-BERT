import pandas as pd
import json
from models.simple_model import SimpleModel

DATA_SOURCE = 'AAPR'

model_list = {'simple_model', 'multi_layer', 'CNN', 'LSTM', 'GRU', 'RNNAttention', 'DPCNN', 'RCNN', 'Transformer', 'BERT', 'SciBERT'}


def get_configs(data_source, model_list):
    fr = open('{}.json'.format(data_source))
    configs = json.load(fr)
    full_configs = {}
    for model in model_list:
        full_configs[model] = configs['default'].copy()
        if model in configs.keys():
            for key in configs[model].keys():
                full_configs[model][key] = configs[model][key]
    print(full_configs['BERT'])
    return full_configs


configs = get_configs(DATA_SOURCE, model_list)
print(configs['simple_model']['wtf'])