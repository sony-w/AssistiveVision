__author__ = 'sony-w'
__version__ = '1.0'

import os
import numpy as np

from tqdm.auto import tqdm

ucb_palette = {
    'berkeley_blue': '#003262',
    'california_gold': '#FDB515',
    'metallic_gold': '#BC9B6A',
    'founders_rock': '#2D637F',
    'medalist': '#E09E19',
    'bay_fog': '#C2B9A7',
    'lawrence': '#00B0DA',
    'sather_gate': '#B9D3B6',
    'pacific': '#53626F',
    'soybean': '#9DAD33',
    'california_purple': '#5C3160',
    'south_hall': '#6C3302',
    'stone_pine': '#584F29',
    'ion': '#CFDD45'
}

def glove_dict(glove_dir, dim=200):
    
    embeddings = dict()
    with open(os.path.join(glove_dir, f'glove.6B.{dim}d.txt'), encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings[words] = coefs
        
    
    return embeddings

def embedding_matrix(embedding_dim, word2idx, glove_dir):
    
    embedding_idx = glove_dict(glove_dir=glove_dir, dim=embedding_size)
    embedding_mtx = np.zeros((len(word2idx), embedding_dim))
    
    for word, i in tqdm(word2idx.items()):
        embedding_vector = embedding_idx.get(word.lower())
        if embedding_vector is not None:
            embedding_mtx[i] = embedding_vector
    
    return embedding_mtx
    
    