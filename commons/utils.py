__author__ = 'sony-w'
__version__ = '1.0'

import os
import itertools
import numpy as np
import torch

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
    """
    Glove embedding of the given dimension.
    
    Returns:
        embeddings(dict): dictionary of the glove embeddings
    """
    
    assert dim in [50, 100, 200, 300], "dim value must be either 50, 100, 200, or 300."
    
    embeddings = dict()
    with open(os.path.join(glove_dir, f'glove.6B.{dim}d.txt'), encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings[word] = coefs
    
    return embeddings


def embedding_matrix(embedding_dim, word2idx, glove_dir):
    """
    Generate embedding matrix from the given tokens (words)
    
    Parameters:
        embedding_dim(int): embedding dimension (length)
        word2idx(dict): dictionary of word to index
        glove_dir(str): glove directory
    
    Returns:
        embedding_mtx(np.array): embedding matrix by (len(word2idx) x embedding_dim)
    """
    
    embedding_idx = glove_dict(glove_dir=glove_dir, dim=embedding_dim)
    embedding_mtx = np.zeros((len(word2idx), embedding_dim))
    
    for word, i in tqdm(word2idx.items()):
        embedding_vector = embedding_idx.get(word.lower())
        if embedding_vector is not None:
            embedding_mtx[i] = embedding_vector
    
    return embedding_mtx
    

def tensor_to_word_fn(idx2word, max_len=40, endseq='<end>'):
    """
    Convert predicted tensors to words
    
    Parameters:
        idx2word(dict): dictionary of index to word
        max_len(int): maximum length
        endseq(str): end of sequence
    """
    
    def tensor_to_word(captions: np.array) -> list:
        
        tokens = []
        for caption in captions:
            tokens.append(list(itertools.takewhile(lambda token: token != endseq, 
                                                   map(lambda idx: idx2words[idx], iter(caption))))[1:])
        return tokens
    
    return tensor_to_word

def accuracy_fn(ignore_value:int=0):
    
    def accuracy_ignoring_value(source:torch.Tensor, target:torch.Tensor):
        mask = target != ignore_value
        return (source[mask] == target[mask]).sum().item() / mask.sum().item()
    
    return accuracy_ignoring_value