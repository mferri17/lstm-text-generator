
# --------------------------------

# %tensorflow_version 1.x
import io
import numpy as np
import tensorflow as tf

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# --------------------------------

# from google.colab import drive
# drive.mount('/content/drive')

# --------------------------------

# with open('drive/My Drive/_ USI/Deep Learning Lab/datasets/montecristo.txt', 'r') as f:
#   book = f.read()

with open('datasets/montecristo.txt', 'r') as f:
  book = f.read()

book = book.lower()

# --------------------------------

### Characters distribution

import pandas
from collections import Counter
from collections import OrderedDict
import string

char_counts = Counter(book)

char_counts_byletter = OrderedDict(sorted(char_counts.items()))
print(f'Characters count ordered alphabetically: {char_counts_byletter}')
df_char_counts_byletter = pandas.DataFrame.from_dict(char_counts_byletter, orient='index')
df_char_counts_byletter.plot(kind='bar')

char_counts_alphabet = dict((i, char_counts_byletter[i]) for i in list(string.ascii_lowercase))
print(f'Alphabet count: {char_counts_alphabet}')
df_char_counts_alphabet = pandas.DataFrame.from_dict(char_counts_alphabet, orient='index')
df_char_counts_alphabet.plot(kind='bar')

top = 20
print(f'Top {top} most common characters')
char_counts.most_common()[:top]

# --------------------------------

#### Handle text to numerical conversion

vocab = sorted(set(book))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def text_to_num(text):
  return np.array([char2idx[c] for c in text])

def num_to_text(nums):
  return ''.join(idx2char[np.array(nums)])


# book = book.lower() # already done before analysis
book_to_num = text_to_num(book)

# --------------------------------

def generate_batches(source, batch_size, sequence_length):
  block_length = len(source) // batch_size

  batches = []
  for i in range(0, block_length, sequence_length):
    batch=[]

    for j in range(batch_size):
      start = j * block_length + i
      end = min(start + sequence_length, j * block_length + block_length)
      batch.append(source[start:end])

    batches.append(np.array(batch, dtype=int))

  return batches

# --------------------------------

#### Model parameters

batch_size = 16
sequence_length = 256
k = len(char_counts) # Input dimension (unique characters in the text)

hidden_units = 256 # Number of recurrent units
learning_rate = 1e-2
n_epochs = 5

# --------------------------------

### Creating dataset for training

bts = generate_batches(book_to_num, batch_size, sequence_length)
print('Number of batches', len(bts)) # ceiling(len(text) / batch_size / sequence_length)
print('Batch size', len(bts[0]))
print('Sequence length', len(bts[0][0]))

# # Just to notice that last batch is incomplete
# for i in range(len(bts)):
#   for j in range(batch_size):
#     if len(bts[i][j]) != 256:
#       print(len(bts[i][j]), i, j)

bts = np.array(bts[:-1]) # removing last batch because incomplete
print('\nbts shape: ' , bts.shape)

data_X = bts
data_Y = np.copy(data_X)

for batch in range(np.shape(bts)[0]):
  for sequence in range(np.shape(bts)[1]):
    for character in range(np.shape(bts)[2] - 1):
      data_Y[batch][sequence][character] = data_X[batch][sequence][character+1]
    data_Y[batch][sequence][np.shape(bts)[2] - 1] = 0 # last character has no target

print('data_X shape: ', data_X.shape)
print('data_Y shape: ', data_Y.shape)

# --------------------------------

### Model definition

seed = 0
tf.reset_default_graph()
tf.set_random_seed(seed=seed)

X_int = tf.placeholder(shape=[None, None], dtype=tf.int64)
Y_int = tf.placeholder(shape=[None, None], dtype=tf.int64)
lengths = tf.placeholder(shape=[None], dtype=tf.int64)

batch_size_tf = tf.shape(X_int)[0]
max_len = tf.shape(X_int)[1] # TODO

# One-hot encoding X_int
X = tf.one_hot(X_int, depth=k) # shape: (batch_size, max_len, k)
# One-hot encoding Y_int
Y = tf.one_hot(Y_int, depth=k) # shape: (batch_size, max_len, k)

# Recurrent Neural Network
basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_units)

# Long-Short Term Memory Neural Network
rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [256, 256]]
multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_units)

init_state = lstm_cell.zero_state(batch_size_tf, dtype=tf.float32)
current_state = lstm_cell.zero_state(batch_size_tf, dtype=tf.float32)

# rnn_outputs shape: (batch_size, max_len, hidden_units)
rnn_outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X, sequence_length=lengths, initial_state=current_state)

# rnn_outputs_flat shape: ((batch_size * max_len), hidden_units)
rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, hidden_units])

# Weights and biases for the output layer
Wout = tf.Variable(tf.truncated_normal(shape=(hidden_units, k), stddev=0.1))
bout = tf.Variable(tf.zeros(shape=[k]))

# Z shape: ((batch_size * max_len), k)
Z = tf.matmul(rnn_outputs_flat, Wout) + bout

Y_flat = tf.reshape(Y, [-1, k]) # shape: ((batch_size * max_len), k)

# Creates a mask to disregard padding
mask = tf.sequence_mask(lengths, dtype=tf.float32)
mask = tf.reshape(mask, [-1]) # shape: (batch_size * max_len)

# Network prediction
pred = tf.squeeze(tf.random.categorical(Z, 1)) * tf.cast(mask, dtype=tf.int64)
pred = tf.reshape(pred, [-1, max_len]) # shape: (batch_size, max_len)

hits = tf.reduce_sum(tf.cast(tf.equal(pred, Y_int), tf.float32))
hits = hits - tf.reduce_sum(1 - mask) # Disregards padding

# Accuracy: correct predictions divided by total predictions
accuracy = hits/tf.reduce_sum(mask)

# Loss definition (masking to disregard padding)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_flat, logits=Z)
loss = tf.reduce_sum(loss*mask)/tf.reduce_sum(mask)

optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# --------------------------------

### Training
print('\n\n --- TRAINING --- \n')

session = tf.Session()
session.run(tf.global_variables_initializer())

batches_number = np.shape(data_X)[0]
losses = np.zeros((n_epochs, batches_number))

for e in range(1, n_epochs + 1):
  cs = session.run(init_state, {X_int: data_X[0], Y_int: data_Y[0]}) # initial state
  
  for b in range(batches_number):
    c_input = data_X[b]
    c_target = data_Y[b]
    ls = list([np.shape(c_input)[1]] * np.shape(c_input)[0])
    feed = {X_int: data_X[b],
            Y_int: data_Y[b],
            lengths: ls, 
            current_state.c: cs.c, 
            current_state.h: cs.h}
    l, _, cs = session.run([loss, train, final_state], feed)
    print(f'Epoch {e}, Batch {b}. \t Loss: {l}')
    losses[e-1][b] = l # saving losses

# --------------------------------

### Loss plot
import matplotlib
import matplotlib.pyplot as plt

print('losses shape', np.shape(losses))

colors = ['#616BB0', '#74C49D', '#FFFF00', '#B02956', '#B3BAFF']

# Total loss
ys = losses.reshape(-1)
xs = np.arange(len(ys))
plt.plot(xs, ys, '-', c=colors[0], label='training loss over all epochs')
plt.legend()
plt.show()

# By epochs
for e in range(len(losses)):
  ys_e = losses[e]
  xs_e = np.arange(len(ys_e))
  plt.plot(xs_e, ys_e, '-', c=colors[0], label=f'training loss (epoch {e+1})')
  plt.legend()
  plt.show()

# By epochs all together
for e in range(len(losses)):
  ys_e = losses[e]
  xs_e = np.arange(len(ys_e))
  plt.plot(xs_e, ys_e, '-', c=colors[e], label=f'training loss (epoch {e+1})')
plt.legend()
plt.show()

# --------------------------------

### Model saving

saver = tf.train.Saver()
saver.save(session, 'models/Activity7Model_1.ckpt')

# --------------------------------

### Text generation

import random
import itertools


for n in range(20):
  ri = random.randrange(sum(char_counts.values()))
  starting_char = next(itertools.islice(char_counts.elements(), ri, None))

  gen_input = [text_to_num(starting_char)] # starting character
  gen_lengths = [1] # generation is done character by character
  cs = session.run(init_state, {X_int: gen_input}) # initial state

  gen_text = [gen_input[0][0]] # store the generated text
  for i in range(255):
    cs, gen_input = session.run([final_state, pred], {X_int: gen_input, lengths: gen_lengths, current_state: cs})
    gen_text.append(gen_input[0][0]) 

  print(f'\n\n------- EXAMPLE {n+1} -------\n')
  print(num_to_text(gen_text))

# --------------------------------

### Restore from model

saver = tf.train.Saver()

with tf.Session() as sess:
  saver.restore(sess, 'models/Activity7Model_1.ckpt')
  print('Model restored.')

  ri = random.randrange(sum(char_counts.values()))
  starting_char = next(itertools.islice(char_counts.elements(), ri, None))

  gen_input = [text_to_num(starting_char)] # starting character
  gen_lengths = [1] # generation is done character by character
  cs = session.run(init_state, {X_int: gen_input}) # initial state

  gen_text = [gen_input[0][0]] # store the generated text
  for i in range(255):
    cs, gen_input = session.run([final_state, pred], {X_int: gen_input, lengths: gen_lengths, current_state: cs})
    gen_text.append(gen_input[0][0]) 

  print(f'\n\n------- EXAMPLE -------\n')
  print(num_to_text(gen_text))

# --------------------------------