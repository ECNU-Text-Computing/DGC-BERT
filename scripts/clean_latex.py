# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 18:23:37 2017

@author: ypc
"""

import re
import os
import codecs
import tarfile
import json as js

# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.tokenize import sent_tokenize
from torchtext.data.utils import get_tokenizer

# english_stopwords = stopwords.words('english')
sentence_sep = 'thelongestsentencesepyep'
word_tokenize = get_tokenizer('basic_english')


def clean_math(string):
    while string.count('$') > 1:
        pos0 = string.find('$')
        pos1 = string.find('$', pos0 + 1)
        string = (string[:pos0] + string[pos1 + 1:]).strip()
    return string


def clean_str(string):
    """
    Input:
        string: One line in a latex file.
    Return：
        string cleaned.
    """

    # Remove mathematical formulas between $$
    string = clean_math(string)

    # Remove "ref" 
    string = re.sub(r'~(.*)}', '', string)
    string = re.sub(r'\\cite(.*)}', '', string)
    string = re.sub(r'\\newcite(.*)}', '', string)
    string = re.sub(r'\\ref(.*)}', '', string)

    # retain sentence cut
    string = string.replace('et al.', 'et al')
    string = string.replace(' pp.', ' pp ')
    string = string.replace('. ', sentence_sep)
    string = string.replace('? ', sentence_sep)
    string = string.replace('! ', sentence_sep)
    string = re.sub(r'\.$', sentence_sep, string)
    string = re.sub(r'\?$', sentence_sep, string)
    string = re.sub(r'!$', sentence_sep, string)

    # Remove stopwords
    texts_tokenized = [word.lower() for word in word_tokenize(string)]
    # texts_filtered_stopwords = [word for word in texts_tokenized if word not in english_stopwords]
    # string = ' '.join(texts_filtered_stopwords)
    texts_filtered_stopwords = [word for word in texts_tokenized if not word.startswith('\\')]
    string = ' '.join(texts_filtered_stopwords)

    # # Cut sentences and remove stopwords
    # sentence_cut = sent_tokenize(string)
    # sents_tokenized = [[word.lower() for word in word_tokenize(sentence)] for sentence in sentence_cut]
    # sents_filtered_stopwords = [[word for word in sent if word not in english_stopwords] for sent in sents_tokenized]
    # sents_joined = [' '.join(sent) for sent in sents_filtered_stopwords]
    # string = sentence_sep.join(sents_joined)

    # string = string.replace(',', '')
    string = string.replace('.', '')
    string = string.replace('?', '')
    string = string.replace('!', '')
    string = string.replace('_', ' ')
    string = string.replace('|', ' ')
    string = string.replace('\'', ' ')
    string = string.replace('\"', ' ')
    string = string.replace('/', ' ')
    string = string.replace('$', ' ')
    string = string.replace('~', ' ')
    string = string.replace('\\', ' ')
    string = string.replace('{', ' ')
    string = string.replace('}', ' ')
    string = string.replace('#', ' ')
    string = string.replace('&', ' ')
    string = string.replace('@', ' ')
    string = string.replace('%', ' ')
    string = string.replace('^', ' ')
    string = string.replace('*', ' ')
    string = string.replace('-', ' ')
    string = string.replace('=', ' ')
    string = string.replace('[', ' ')
    string = string.replace(']', ' ')
    string = string.replace('+', ' ')
    # string = string.replace('(', '')
    # string = string.replace(')', '')
    return string + ' '


def process_text_list(text_list):
    """
    Input:
        text_list: Content of a latex file and each element represents a line.
    Return:
        A list, which is the cleaned content of a latex file.
    """

    result = ''
    for line in text_list:
        line = line.strip()
        if line.startswith('%') or line.startswith('\\') or line.startswith('}') or line == '':
            # print(line)
            pass
        elif line[0].isdigit():
            pass
        else:
            result += clean_str(line)
    return result.split(sentence_sep)


# Extract Introduction, related work, etc.================================================================
def split(tex_list, start_char, end_char):
    lines = tex_list
    length = len(lines)
    start = None
    end = None
    i = 0
    while i < length and (end is None):
        if start is None:
            if lines[i].strip().startswith(start_char):
                start = i + 1
        else:
            if lines[i].strip().startswith(end_char) or lines[i].strip().startswith('\\bibitem'):
                end = i
        i += 1
    if (start is not None) and (end is None):
        end = length
    elif (start is None) and (end is None):
        start = end = 0
    # print(start, end)
    return [start, end]


def extract(tex_list, segment=False):
    data = tex_list
    # abstract = ' '.join(split(tex_list, r'\\begin{abstract', r'\\end{'))
    abstract_search = re.search('(\{abstract}\s+)(.*)(\{abstract})', ' '.join(data), re.S)
    abstract = abstract_search.group(2) if abstract_search else ''
    # intro = ' '.join(split(tex_list, '\section{Intro', '\section{'))
    # related = ' '.join(split(tex_list, '\section{Related', '\section{'))
    # conclusion = ' '.join(split(tex_list, '\section{Conclu', '\section{'))
    intro_range = split(tex_list, '\section{Intro', '\section{')
    intro = ' '.join(tex_list[intro_range[0]:intro_range[1]])

    related_range = split(tex_list, '\section{Relate'
                                    'd', '\section{')
    background_range = split(tex_list, '\section{Background', '\section{')
    related_start, related_end = get_range(related_range, background_range)
    related = ' '.join(tex_list[related_start:related_end])

    conclusion_range = split(tex_list, '\section{Conclu', '\section{')
    discussion_range = split(tex_list, '\section{Discuss', '\section{')
    conclusion_start, conclusion_end = get_range(conclusion_range, discussion_range)
    conclusion = ' '.join(tex_list[conclusion_start:conclusion_end])
    end = len(data) if conclusion_start == 0 else conclusion_start
    methods = ' '.join(tex_list[intro_range[0]:end]).replace(abstract, '').replace(intro, '').replace(related,
                                                                                                      '').replace(
        conclusion, '')
    if segment:
        pass
    else:
        return list(map(process_text_list,
                        [abstract.split('\n'), intro.split('\n'), related.split('\n'), methods.split('\n'),
                         conclusion.split('\n')]))


# def main(file_dir):
#     result = {}
#     file_names = os.listdir(file_dir)
#     for file_name in file_names:
#         try:
#             f_name = os.path.join(file_dir, file_name)
#             tex_list = make_single_tex(f_name)
#             result[file_name] = extract(tex_list)
#         except:
#             continue
#     return result

def get_range(section_1, section_2):
    if section_1[0] != 0 and section_2 != 0:
        start = min(section_1[0], section_2[0])
    else:
        start = max(section_1[0], section_2[0])
    end = max(section_1[1], section_2[1])
    return [start, end]