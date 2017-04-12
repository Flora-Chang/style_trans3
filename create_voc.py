"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys

from tensorflow.python.platform import gfile
import pprint, pickle
import numpy as np

# Special vocabulary symbols - we always put them at the start.
#from tf_seq2seq_chatbot.configs.config import BUCKETS

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
UNknown = "UNknown"
_START_VOCAB = [UNknown]

#PAD_ID = 0
#GO_ID = 1
#EOS_ID = 1
#UNK_ID = 2
UNK_ID = 1
vocab_size =25000

# Regular expressions used to tokenize.
'''
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d{3,}")
'''

def decode(path, new_path):
    pkl_file = open(path, 'rb')
    data = pickle.load(pkl_file, encoding='utf-8')
    f = open(new_path, 'w')
    for line in data:
        seq = ''
        for word in line:
            seq = seq+' ' +str(word)
        f.write(seq.strip() + '\n')
    f.close()
def create_vocabulary(vocabulary_path, data_path_news, data_path_paper, max_vocabulary_size,):
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s, %s" % (vocabulary_path, data_path_news, data_path_paper))
    vocab = {}
    pkl_file_news = open(data_path_news, 'rb')
    data_news = pickle.load(pkl_file_news, encoding='utf-8')
    counter =0
    for line in data_news:
        counter += 1
        if counter % 10000 ==0:
            print("  processing line %d" % counter)
        for word in line:
            if word in vocab:
                vocab[word] += 1
            else :
                vocab[word] = 1
    pkl_file_paper = open(data_path_paper, 'rb')
    data_paper = pickle.load(pkl_file_paper, encoding='utf-8')
    counter = 0
    for line in data_paper:
        counter += 1
        if counter % 10000 == 0:
            print("  processing line %d" % counter)
        for word in line:
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size+1]
    print (type(vocab_list))
    with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
        for w in vocab_list:
            vocab_file.write(w + "\n")

def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.
  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].
  Args:
    vocabulary_path: path to the file containing the vocabulary.
  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).
  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []

    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())

    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict((x, y+1) for (y, x) in enumerate(rev_vocab))
    return vocab, rev_vocab

  else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def data_to_token_ids(data_path, target_path, vocabulary_path):
    """
    Tokenize data file and turn into token-ids using given vocabulary file.
    """
    if not gfile.Exists(target_path):
        word2id = []
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        pkl_file = open(data_path, 'rb')
        data = pickle.load(pkl_file, encoding ='utf-8')
        #print (data[:10])
        with gfile.GFile(target_path, mode="w") as tokens_file:
            counter = 0
            for i,line in enumerate(data):
                token_ids = []
                counter += 1
                if counter % 10000 == 0:
                    print("  processing line %d" % counter)
                for j,word in enumerate(line):
                    token_ids.append(vocab.get(word, UNK_ID))
                #print(token_ids)
                #word2id.append(token_ids)
                tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
                #tokens_file.write(" ".join(word for word in line) + '\n')
        #return word2id
def concat_paper_news(paper_path, news_path, target_path):
    with open(paper_path, 'r') as fp, open(news_path, 'r') as fn, open(target_path, 'w') as ft:
        content_p = fp.readlines()
        content_n = fn.readlines()
        len_p = len(content_p)
        len_n = len(content_n)
        print('size paper and news %d and %d:', len_p, len_n)

        for i, (p,n ) in enumerate(zip(content_p, content_n)):
            p = p.strip()
            n = n.strip()
            str =p + '|' + n
            ft.write(str+'\n')
def data_split(alldata_path, train_path, val_path, test_path, rate_train, rate_val):
    with open(alldata_path, 'r') as fa, open(train_path, 'w') as ftr, open(val_path,'w') as fv, open(test_path, 'w') as fte:
        allcontent = fa.readlines()
        np.random.shuffle(allcontent)
        all_len = len(allcontent)
        for i, content in enumerate(allcontent):
            content = content.strip()
            if i<all_len*rate_train:
                ftr.write(content +'\n')
            if i>=all_len*rate_train and i< all_len*(rate_train+rate_val):
                fv.write(content +'\n')
            if i>all_len*(rate_train+rate_val) and i<all_len:
                fte.write(content+'\n')
def data_split1(alldata_path, train_path, val_path, test_path, rate):
    with open(alldata_path, 'r') as fa, open(train_path, 'w') as ftr, open(val_path,'w') as fv, open(test_path, 'w') as fte:
        allcontent = fa.readlines()
        np.random.shuffle(allcontent)
        all_len = len(allcontent)
        for i, content in enumerate(allcontent):
            content = content.strip()
            if i<all_len*rate:
                ftr.write(content +'\n')
            if i>=all_len*rate and i< all_len*(rate)*2:
                fv.write(content +'\n')
            if i>all_len*(rate*2) and i<all_len*(rate*3):
                fte.write(content+'\n')
def decode_token_to_id(input_path, output_path, voc):
    with open(input_path, 'r', encoding = 'utf-8') as fi, open(output_path, 'w', encoding='utf-8') as fo:
        cnt = 0
        for line in fi.readlines():
            strings = line.strip().split('|')
            if len(strings)!=2:
                cnt+=1
                print(line)
                print("strings len is not 2")
            out_sequence = ''
            for i,sequence in enumerate(strings):
                if i==2:
                    break
                for token in sequence.split(' '):
                    token = str(token)
                    if token in voc.keys():
                        token = str(voc[token])
                    else:
                        token = '1'
                    out_sequence=out_sequence + token + ' '
                if i ==0:
                    out_sequence = out_sequence.strip()+'|'
            fo.write(out_sequence+'\n')
        print (cnt)

def decode_id_to_token(input_path, output_path, rev_voc):
    with open(input_path, 'r') as fi, open(output_path, 'w') as fo:
        for line in fi.readlines():
            strings = line.strip().split('|')
            if len(strings)!=2:
                print("strings len is not 2")
            out_sequence = ''
            for i,sequence in enumerate(strings):
                for id in sequence.split(' '):
                    id=int(id)
                    out_sequence=out_sequence + rev_voc[id-1] +' '
                if i ==0:
                    out_sequence = out_sequence.strip()+'|'
            fo.write(out_sequence+'\n')



def main():
    '''
    decode('./title/news_pa','./title/news_p.txt')
    decode('./title/paper_pa', './title/paper_p.txt')
    
    create_vocabulary('./data/movie_25000', './title/news_pa','./title/paper_pa', vocab_size)
    vocab, rev_vocab = initialize_vocabulary('./data/movie_25000')
    data_to_token_ids('./title/news_pa', './data/word2id_news.txt','./data/movie_25000')
    data_to_token_ids('./title/paper_pa', './data/word2id_paper.txt', './data/movie_25000')
    concat_paper_news('./data/word2id_paper.txt', './data/word2id_news.txt', './data/t_given_s_all.txt')
    data_split('./data/t_given_s_all.txt','./data/t_given_s_train.txt','./data/t_given_s_dev.txt','./data/t_given_s_test.txt', 0.7,0.15)
    #data_split1('./data/t_given_s_all.txt','./data/t_given_s_train.txt','./data/t_given_s_dev.txt','./data/t_given_s_test.txt', 0.15)
    print(len(vocab))
    print(rev_vocab[:10])
    print(len(rev_vocab))
    #print(word2id[:10])
    
    #vocab, rev_vocab=create_vocabulary('./data/test.txt', './title/news_pa','./title/paper_pa', vocab_size)
    vocab, rev_vocab = initialize_vocabulary('./data/movie_25000')
    print(type(vocab))
    print (vocab['UNknown'])
    #print(vocab["\'like"])
    print(rev_vocab[:10])
    print(rev_vocab[23397])
    print(len(rev_vocab))
    print(len(vocab))
    print(vocab['mycobiome'])
    #print (vocab.keys())
    #print(rev_vocab)
    '''
    vocab, rev_vocab = initialize_vocabulary('./data/movie_25000')
    decode_token_to_id("./data/decode_dev_iter1.txt", "./decode_dev_iter1.txt", vocab)
    decode_token_to_id("./data/decode_test_iter1.txt", "./decode_test_iter1.txt", vocab)
    decode_token_to_id("./data/decode_train_iter1.txt", "./decode_train_iter1.txt", vocab)




    #decode_id_to_token('./data/t_given_s_dev.txt', './data/decode_dev_real.txt', rev_vocab)
    #decode_id_to_token('./data/t_given_s_test.txt', './data/decode_test_real.txt', rev_vocab)
    #decode_id_to_token('./data/t_given_s_train.txt', './data/decode_train_real.txt', rev_vocab)

    #decode_token_to_id('decode_dev.txt', 'decode_dev_data.txt', vocab)

if __name__ == '__main__':
    main()
