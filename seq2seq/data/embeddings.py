'''
Created on Jul 4, 2017

@author: Bhanu

'''
import numpy as np
from seq2seq.data import vocab
import tensorflow as tf

def read_embeddings(embeddings_path, vocab_path):
  """Reads embeddings file.

  Args:
    embeddings_path: full path for the embeddings file,
      where embeddings file contains word and its vector(float values)
      per line, separated by blank space.
    vocab_path: full path for the vocab file,
      where each line contains a single vocab word.

  Returns:
    a 2d array where row index corresponds to the word index
    in the vocab (special vocab and other unknown words are
    also included at their respective row index.
  """
  vocab_, _, _ = vocab.read_vocab(vocab_path)
  word2vec = {}
  with open(embeddings_path, 'r') as vec_file:
    for line in vec_file:
      parts = line.split(' ')
      word = parts[0]
      emb_dim = len(parts) - 1
      if word not in vocab_: continue

      vec = parts[1:]
      word2vec[word] = vec

  unknown_words = [w for w in vocab_  if w not in word2vec]
  rnd_vecs = [np.random.uniform(-0.25, 0.25, size=emb_dim)
              .tolist() for _ in unknown_words]
  tf.logging.info("adding %d unknown words to vocab", len(unknown_words))
  word2vec.update(dict(zip(unknown_words, rnd_vecs)))

  vecs = [word2vec.get(w) for w in vocab_]
  embedding_mat = np.asarray(vecs, dtype=np.float32)

  return embedding_mat
