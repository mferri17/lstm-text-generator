
# --------------------------------

#%tensorflow_version 1.x
import numpy as np
import tensorflow as tf
import sys

# --------------------------------

# from google.colab import drive
# drive.mount('/content/drive')

# --------------------------------

# with open('drive/My Drive/_ USI/Deep Learning Lab/datasets/montecristo.txt', 'r') as f:
#   book = f.read()

with open('datasets/montecristo.txt', 'r') as f:
  book = f.read()

book = book.lower()
print(book)

# --------------------------------

import pandas
from collections import Counter
from collections import OrderedDict

char_counts = Counter(book)

# --------------------------------

char_counts_byletter = OrderedDict(sorted(char_counts.items()))
print(f'Characters count ordered alphabetically: {char_counts_byletter}')
df_char_counts_byletter = pandas.DataFrame.from_dict(char_counts_byletter, orient='index')
df_char_counts_byletter.plot(kind='bar')

top = 15
print(f'Top {top} most common characters')
char_counts.most_common()[:top]

# --------------------------------

def text_to_num(text):
  return list(map(lambda x: ord(x), text))

# --------------------------------

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

example_text = 'Mi chiamo Marco e sono un gattino.'
example_num = text_to_num(example_text)
print(example_num)
print(generate_batches(example_num, 3, 2))

# --------------------------------

batch_size = 16
sequence_length = 256
bts = generate_batches(book_to_num, batch_size, sequence_length)
# print(len(book_to_num) / batch_size / sequence_length)
print('Number of batches', len(bts)) # ceiling(len(text) / batch_size / sequence_length)
print('Batch size',len(bts[0]))
print('Sequence length',len(bts[0][0]))

# --------------------------------

for i in range(len(bts)):
  for j in range(batch_size):
    if len(bts[i][j]) != 256:
      print(len(bts[i][j]), i, j)

# --------------------------------

np.array(bts[:-1]).shape

# --------------------------------

seed = 0
tf.reset_default_graph()
tf.set_random_seed(seed=seed)

# Task parameters
# TODO
n = 3 # n-back
k = 4 # Input dimension
mean_length = 20 # Mean sequence length
std_length = 5 # Sequence length standard deviation
n_sequences = 512 # Number of training/validation sequences

# Creating datasets
random_state = np.random.RandomState(seed=seed)
X_train, Y_train, lengths_train = nback_dataset(n_sequences, mean_length, std_length, n, k, random_state) # TODO

X_val, Y_val, lengths_val = nback_dataset(n_sequences, mean_length, std_length, n, k, random_state)  # TODO

# Model parameters
hidden_units = 256 # Number of recurrent units

# Training procedure parameters
learning_rate = 1e-2
n_epochs = 5

# --------------------------------

# Model definition
X_int = tf.placeholder(shape=[None, None], dtype=tf.int64)
Y_int = tf.placeholder(shape=[None, None], dtype=tf.int64)
lengths = tf.placeholder(shape=[None], dtype=tf.int64)

batch_size = tf.shape(X_int)[0]
max_len = tf.shape(X_int)[1]

# One-hot encoding X_int
X = tf.one_hot(X_int, depth=k) # shape: (batch_size, max_len, k)
# One-hot encoding Y_int
Y = tf.one_hot(Y_int, depth=2) # shape: (batch_size, max_len, 2)

# Recurrent Neural Network
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_units)

# Long-Short Term Memory Neural Network
# cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_units)

init_state = cell.zero_state(batch_size, dtype=tf.float32)


# rnn_outputs shape: (batch_size, max_len, hidden_units)
rnn_outputs, \
    final_state = tf.nn.dynamic_rnn(cell, X, sequence_length=lengths, initial_state=init_state)

# rnn_outputs_flat shape: ((batch_size * max_len), hidden_units)
rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, hidden_units])

# Weights and biases for the output layer
Wout = tf.Variable(tf.truncated_normal(shape=(hidden_units, 2), stddev=0.1))
bout = tf.Variable(tf.zeros(shape=[2]))

# Z shape: ((batch_size * max_len), 2)
Z = tf.matmul(rnn_outputs_flat, Wout) + bout

Y_flat = tf.reshape(Y, [-1, 2]) # shape: ((batch_size * max_len), 2)

# Creates a mask to disregard padding
mask = tf.sequence_mask(lengths, dtype=tf.float32)
mask = tf.reshape(mask, [-1]) # shape: (batch_size * max_len)

# Network prediction
pred = tf.argmax(Z, axis=1) * tf.cast(mask, dtype=tf.int64)
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

session = tf.Session()
session.run(tf.global_variables_initializer())

for e in range(1, n_epochs + 1):
    feed = {X_int: X_train, Y_int: Y_train, lengths: lengths_train}
    l, _ = session.run([loss, train], feed)
    print(f'Epoch: {e}. Loss: {l}.')

feed = {X_int: X_val, Y_int: Y_val, lengths: lengths_val}
accuracy_ = session.run(accuracy, feed)
print(f'Validation accuracy: {accuracy_}.')

# Shows first task and corresponding prediction
xi = X_val[0, 0: lengths_val[0]]
yi = Y_val[0, 0: lengths_val[0]]
print('Sequence:')
print(xi)
print('Ground truth:')
print(yi)
print('Prediction:')
print(session.run(pred, {X_int: [xi], lengths: [len(xi)]})[0])

session.close()

# --------------------------------