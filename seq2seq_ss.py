import numpy as np
import tensorflow as tf
import helpers
import time
#from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import create_voc
#config = tf.ConfigProto(allow_soft_placement=True)
tf.reset_default_graph()
sess = tf.InteractiveSession()
input_news = helpers.input_list('./word2id_news.txt')
input_paper = helpers.input_list('./word2id_paper.txt')
PAD = 0
EOS = 1
vocab_size = 50000
input_embedding_size = 1000

encoder_hidden_units = 1000
decoder_hidden_units = encoder_hidden_units * 2
#with tf.device('/gpu:0'):
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
decoder_targets_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_targets_length')

decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
# W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
# b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)

encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

encoder_cell = LSTMCell(encoder_hidden_units)
((encoder_fw_outputs,
  encoder_bw_outputs),
 (encoder_fw_final_state,
  encoder_bw_final_state)) = (
    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                    cell_bw=encoder_cell,
                                    inputs=encoder_inputs_embedded,
                                    sequence_length=encoder_inputs_length,
                                    dtype=tf.float32, time_major=True)
)
print(encoder_fw_outputs, encoder_bw_outputs, encoder_fw_final_state, encoder_bw_final_state)

encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

encoder_final_state_c = tf.concat(
    (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

encoder_final_state_h = tf.concat(
    (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

encoder_final_state = LSTMStateTuple(
    c=encoder_final_state_c,
    h=encoder_final_state_h
)

decoder_cell = LSTMCell(decoder_hidden_units)
encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))

decoder_lengths = decoder_targets_length
W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)

assert EOS == 1 and PAD == 0

eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)


def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
    initial_input = eos_step_embedded
    initial_cell_state = encoder_final_state
    initial_cell_output = None
    initial_loop_state = None  # we don't need to pass any additional information
    return (initial_elements_finished,
            initial_input,
            initial_cell_state,
            initial_cell_output,
            initial_loop_state)


def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
    def get_next_input():
        output_logits = tf.add(tf.matmul(previous_output, W), b)
        prediction = tf.argmax(output_logits, axis=1)
        next_input = tf.nn.embedding_lookup(embeddings, prediction)
        return next_input

    elements_finished = (time >= decoder_lengths)  # this operation produces boolean tensor of [batch_size]
    # defining if corresponding sequence has ended

    finished = tf.reduce_all(elements_finished)  # -> boolean scalar
    input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
    state = previous_state
    output = previous_output
    loop_state = None

    return (elements_finished,
            input,
            state,
            output,
            loop_state)


def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_state is None:  # time == 0
        assert previous_output is None and previous_state is None
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)


decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_outputs_ta.stack()

print('decoder_output:', decoder_outputs)

decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))

decoder_prediction = tf.argmax(decoder_logits, 2)

stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)

loss = tf.reduce_mean(stepwise_cross_entropy)

train_op = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#train_op = tf.train.AdamOptimizer().minimize(loss)
#train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(loss)

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

batch_size = 100
#  an Iterater
l = len(input_news)
#index list
trains, val, test = helpers.datasplit(l)
#batches_train = helpers.batch_sequences(trains, batch_size=batch_size)
#batches_val = helpers.batch_sequences(val, batch_size=batch_size)
#batches_test = helpers.batch_sequences(test, batch_size=batch_size)

def next_feed(batches):
    #batch_index = next(batches)
    batch_index = []
    j=0
    for i in batches:
        batch_index.append(i)
        j+=1
        if j >=100:
            batch_news = [input_news[k] for k in batch_index]
            batch_paper = [input_paper[k] for k in batch_index]
            encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch_paper)
            decoder_targets_, decoder_targets_lengths_ = helpers.batch(
                [(sequence) + [EOS] + [PAD]  for sequence in batch_news]
            )
            batch_index = []
            j=0
            yield {
                encoder_inputs: encoder_inputs_,
                encoder_inputs_length: encoder_input_lengths_,
                decoder_targets_length: decoder_targets_lengths_,
                decoder_targets: decoder_targets_,
            }

save_epoch = 1
max_epoch = 100
def train():
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    vocab, rev_vocab = create_voc.initialize_vocabulary('./voc.txt')
    train_loss_track = []
    val_loss_track = []
    start_time = time.time()
    saver.restore(sess, "./save/var.chkp")
    try:
        for epoch in range(max_epoch):
            np.random.shuffle(trains)
            #print(trains[:3])
            #batches_train = helpers.batch_sequences(trains, batch_size=batch_size)
            num=0
            for fd in next_feed(trains):
                print("minibach:" , num)
                #print(fd)
                num+=1
                _, l = sess.run([train_op, loss], fd)
                print(l)
            if epoch==0 or epoch % save_epoch == 0:
                saver.save(sess, "./save/var.chkp")
                print('epoch {}'.format(epoch))
                print('train  minibatch loss: {}'.format(sess.run(loss, fd)))
                train_loss_track.append(sess.run(loss, fd))
                predict_ = sess.run(decoder_prediction, fd)
                for i, (inp, oup, pred) in enumerate(zip(fd[encoder_inputs].T, fd[decoder_targets].T, predict_.T)):
                    pres = helpers.output_transfer(rev_vocab, pred)
                    ins = helpers.output_transfer(rev_vocab, inp)
                    targs = helpers.output_transfer(rev_vocab, oup)
                    print('  sample {}:'.format(i + 1))
                    print('    paper    > {}'.format(ins))
                    print('    targets_news > {}'.format(targs))
                    print('    news > {}'.format(pres))
                    with open('./train_prediction.txt', 'a') as f:
                        f.write('epoch: ' + str(epoch) +'\n')
                        f.write('paper: '+ inp+ + '\n')
                        f.write('targets_news: ' + targs+'\n')
                        f.write('pre_news: ' + pres +'\n')
                    if i >= 2:
                        break
                print()
                sum = 0
                j = 0
                np.random.shuffle(val)
                #batches_val = helpers.batch_sequences(val, batch_size=batch_size)
                for fd_val in next_feed(val):
                    j+=1
                    val_loss = sess.run(loss, fd_val)
                    sum += val_loss
                    if j==3:
                        break
                print('validation minibatch loss: {}'.format(float(sum / 3)))
                val_loss_track.append(float(sum / 3))
                predict_val = sess.run(decoder_prediction, fd_val)
                for i, (inp, oup, pred) in enumerate(
                        zip(fd_val[encoder_inputs].T, fd_val[decoder_targets].T, predict_val.T)):
                    pres = helpers.output_transfer(rev_vocab, pred)
                    ins = helpers.output_transfer(rev_vocab, inp)
                    targs = helpers.output_transfer(rev_vocab, oup)
                    print('  sample {}:'.format(i + 1))
                    print('    paper    > {}'.format(ins))
                    print('    targets_news > {}'.format(targs))
                    print('    news > {}'.format(pres))
                    with open('./val_prediction.txt', 'a') as f:
                        f.write('epoch: ' + str(epoch) +'\n')
                        f.write('paper: '+ ins+'\n')
                        f.write('targets_news: ' + targs+'\n')
                        f.write('pre_news: ' + pres +'\n')
                    if i >= 2:
                        break
                with open('./train_loss.txt', 'a', encoding='utf-8') as t_f:
                    for l in train_loss_track:
                        t_f.write(str(l) + '\n')

                with open('./val_loss.txt', 'a', encoding='utf-8') as v_f:
                    for l in val_loss_track:
                        v_f.write(str(l) + '\n')
    except KeyboardInterrupt:
        print('training interrupted')
    print('sum_time:', time.time() - start_time)

def main():
    train()

if __name__=='__main__':
    main()