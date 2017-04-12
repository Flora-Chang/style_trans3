import numpy as np
from tensorflow.python.platform import gfile
import random
import create_voc
#translate the input file to list of lists
def input_list(input_path):
    data_all = []
    with gfile.GFile(input_path, mode="r") as input_file:
        for line in input_file.readlines():
            setence = []
            for word in line.strip().split():
                setence.append(word)
            data_all.append(setence)
    return data_all


def batch(inputs, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used

    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active
            time steps in each input sequence
    """

    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # == PAD

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths

'''
def batch_sequences(batches, batch_size):
    """ Generates batches of  sequences
    """
    l =len(batches)
    q = 0
    while q<l:
        yield [
            batches[i+q] for i in range(batch_size)
            ]
        q+=batch_size
'''
'''
def batch_sequences(batches_news,batches_paper, batch_size):
    """ Generates batches of  sequences,news and paper
    """
    l =len(batches_news)
    while True:
        index_list = []
        for i in range(batch_size):
            index_list.append(np.random.randint(0, l))
        #print('index_list:')
        #print(index_list)
        #yield [[batches_news[k] for k in index_list], [batches_paper[k] for k in index_list]]
        yield index_list

'''
def batch_sequences(all_index, batch_size):
    for i in all_index:
        index_list = []
        for j in range(batch_size):
            index_list.append(i)
        yield index_list
'''
def id_to_token(vocb_path):
    with open(vocb_path, 'w') as f:
        id2word = {}
        for id, token in enumerate(f.readlines()):
            id2word[id]=token.strip()
    return id2word
'''
'''translate id to tokens to output'''
def output_transfer(id2word, ids):
    tokens = ''
    indexs = list(ids)
    for id in indexs:
        #print (id)
        id = int(id)
        if id!=0:
            tokens+= id2word[id]+' '
    return tokens
'''split the data into training set and testing set (index list)'''
def datasplit(cnt):
    train = []
    val = []
    test = []
    train_cnt = int(cnt*0.7)
    val_cnt = int(cnt*0.15)
    for i in range(train_cnt):
        train.append(i)
    for j in range(train_cnt, train_cnt+val_cnt):
        val.append(j)
    for k in range(train_cnt+val_cnt, cnt):
        test.append(k)
    return train, val, test
'''
def main():
    vocab, rev_vocab = create_voc.initialize_vocabulary('./voc.txt')
    print(len(vocab))
    print(len(rev_vocab))
    print(rev_vocab[4123])
    print(vocab['analytic'])
    print(vocab['believe'])
    print(rev_vocab[4112])
    in_list_news= input_list('./word2id_news.txt')
    in_list_paper = input_list('./word2id_paper.txt')
    inputs_time_major_n, sequence_lengths_n = batch(in_list_news)
    inputs_time_major_p, sequence_lengths_p = batch(in_list_paper)
    print (len(in_list_news))
    print (len(in_list_paper))
    print (np.shape(inputs_time_major_n), np.shape(sequence_lengths_n))
    print (np.shape(inputs_time_major_p), np.shape(sequence_lengths_p))
    batches =batch_sequences(in_list_news,in_list_paper, 3)

    for i in range(10):
        print (i)
        batch_index = next(batches)
        tmp_news = [in_list_news[k] for k in batch_index]
        tmp_paper = [in_list_paper[k] for k in batch_index]
        print(tmp_news)
        print(tmp_paper)

        for i,(seq_news, seq_paper) in enumerate(zip(tmp_news, tmp_paper)):
            out_news = output_transfer(rev_vocab, seq_news)
            out_paper = output_transfer(rev_vocab, seq_paper)
            print  ('out_news')
            print(out_news)
            print('out_paper')
            print(out_paper)

if __name__ == '__main__':
    main()
'''
