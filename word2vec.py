# -*- coding:utf-8 -*-

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile
import argparse
import pickle

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Step 1: Download the data.
# url = 'http://mattmahoney.net/dc/'
# 
# 
# def maybe_download(filename, expected_bytes):
#   """Download a file if not present, and make sure it's the right size."""
#   if not os.path.exists(filename):
#     filename, _ = urllib.request.urlretrieve(url + filename, filename)
#   statinfo = os.stat(filename)
#   if statinfo.st_size == expected_bytes:
#     print('Found and verified', filename)
#   else:
#     print(statinfo.st_size)
#     raise Exception(
#         'Failed to verify ' + filename + '. Can you get to it with a browser?')
#   return filename

# filename = maybe_download('text8.zip', 31344016)


# # Read the data into a list of strings.
# def read_data(filename):
#   """Extract the first file enclosed in a zip file as a list of words"""
#   with zipfile.ZipFile(filename) as f:
#     data = tf.compat.as_str(f.read(f.namelist()[0])).split()
#   return data
parser = argparse.ArgumentParser(description='XXXXXX')
parser.add_argument('--data', type=str, default='',
                    help='location of the data corpus')
parser.add_argument('--lamb', type=float, default=0.01,
                    help='lambda')
parser.add_argument('--split', type=float, default=0.2,
                    help='split')
parser.add_argument('--lr_rate', type=float, default=1.0,
                    help='lr_rate')
parser.add_argument('--vocab_size', type=float, default=30000,
                    help='vocab_size')
parser.add_argument('--window_size', type=int, default=5,
                    help='window_size')
parser.add_argument('--batch_size', type=int, default=1000,
                    help='batch_size')
parser.add_argument('--emb_size', type=int, default=300,
                    help='emb_size')
parser.add_argument('--adv_hid_size', type=int, default=450,
                    help='adv_hid_size')
parser.add_argument('--adv_batch_size', type=int, default=3000,
                    help='adv_batch_size')
parser.add_argument('--steps', type=int, default=1000000,
                    help='step')
args = parser.parse_args()

def read_data():
    """
    对要训练的文本进行处理，最后把文本的内容的所有词放在一个列表中
    """

    # 读取文本，预处理，分词，得到词典
    raw_word_list = []
    f = open(args.data, 'r').readlines()
    for item in f:
        line = item.strip().split(' ')
        raw_word_list.extend(line)
    return raw_word_list
words = read_data()
print('Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = len(collections.Counter(words).most_common())+1


def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common())
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
fr = open('word', 'w')
fr.write('\n'.join([word for word, _ in count]))
fr.close()
with open('dictionary.pkl', 'wb') as file:
  pickle.dump(dictionary, file)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=args.window_size)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = args.batch_size  
embedding_size = args.emb_size  # Dimension of the embedding vector.
skip_window = args.window_size       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()
# saver = tf.train.Saver()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
  adv_index = tf.placeholder(tf.int32, shape=[args.adv_batch_size])
  sp = int(args.adv_batch_size * args.split)
  y = np.array([[0, 1]] * sp + [[1, 0]] * (args.adv_batch_size - sp)).astype(np.float32) 

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -0.1, 0.1))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    adv_samples = tf.nn.embedding_lookup(embeddings, adv_index)
    adv_weights = tf.Variable(tf.random_uniform([args.adv_hid_size, embedding_size], -0.1, 0.1))
    adv_bias = tf.Variable(tf.random_uniform([args.adv_hid_size, 1], -0.1, 0.1))
    adv_bin_weights = tf.Variable(tf.random_uniform([args.adv_hid_size, 2], -0.1, 0.1))
    adv_bin_bias = tf.Variable(tf.random_uniform([2, 1], -0.1, 0.1))
    
    adv_tmp = tf.matmul(adv_weights, tf.transpose(adv_samples)) + adv_bias
    adv_output = tf.matmul(tf.transpose(adv_tmp), adv_bin_weights) + tf.transpose(adv_bin_bias)
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=adv_output, targets=y)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  raw_loss = tf.reduce_mean(
      tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                     num_sampled, vocabulary_size))

  loss = raw_loss #+ args.lamb * tf.reduce_mean(xentropy)

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(args.lr_rate).minimize(loss)
  # optimizer = tf.train.AdamOptimizer(0.3).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = args.steps

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print("Initialized")

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    sp = int(args.adv_batch_size * args.split)
    sp_total = int(vocabulary_size * args.split)
    samples = np.random.randint(0, sp_total, [sp]).tolist() + \
    np.random.randint(sp_total, vocabulary_size, [args.adv_batch_size - sp]).tolist()
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels, adv_index: samples}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, raw_loss], feed_dict=feed_dict)
    average_loss += loss_val
#     print('step:', step)
    if step % 500 == 0:
      if step > 0:
        average_loss /= 500
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0
      # saver_path = saver.save(sess, "save/model.ckpt")  # 将模型保存到save/model.ckpt文件
      # print("Model saved in file:", saver_path)

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 1000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
      final_embeddings = normalized_embeddings.eval()

      with open('embeddings.pkl', 'wb') as file:
        pickle.dump(final_embeddings, file)
# def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
#   assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
#   plt.figure(figsize=(18, 18))  # in inches
#   for i, label in enumerate(labels):
#     x, y = low_dim_embs[i, :]
#     plt.scatter(x, y)
#     plt.annotate(label,
#                  xy=(x, y),
#                  xytext=(5, 2),
#                  textcoords='offset points',
#                  ha='right',
#                  va='bottom')

#   plt.savefig(filename)

# try:
#   from sklearn.manifold import TSNE
#   import matplotlib.pyplot as plt

#   tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#   plot_only = 500
#   low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
#   labels = [reverse_dictionary[i] for i in xrange(plot_only)]
#   plot_with_labels(low_dim_embs, labels)

# except ImportError:
# print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
