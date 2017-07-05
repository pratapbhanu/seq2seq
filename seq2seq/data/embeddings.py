'''
Created on Jul 4, 2017

@author: Bhanu

'''
import numpy as np
from seq2seq.data import vocab


def read_embeddings(embeddings_path, vocab_path):
  vocab_, _, _ = vocab.read_vocab(vocab_path)   
  word2vec = {}
  with open(embeddings_path, 'rt', encoding='utf-8') as vec_file:
    for line in vec_file:
      parts = line.split(',')
      word = parts[0]

      if word not in vocab_: continue

      vec = parts[1:]
      word2vec[word] = vec
        
  unknown_words = [w for w in vocab_  if w not in word2vec]
  emb_dim = len(word2vec.get(vocab_[0]))
  rnd_vecs = [np.random.uniform(-0.25, 0.25, 
                size=emb_dim).tolist() for _ in unknown_words]
  print("adding %d unknown words to vocab"%len(unknown_words))
  word2vec.update(dict(zip(unknown_words,rnd_vecs)))

  vecs = [word2vec.get(w) for w in vocab_]
  embedding_mat = np.asarray(vecs, dtype=np.float32)

  return embedding_mat


