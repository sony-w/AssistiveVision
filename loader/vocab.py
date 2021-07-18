import pandas as pd
import numpy as np

from collections import Counter


def construct_vocab(self, *args, remove_punctuation=True, threshold=5):
    """
    Generate vocabulary object from all the captions
    Parameters:
        *args(pd.DataFrame): tokens dataframe
        remove_punctuation(boolean): flag to remove punctuation
        threshold(int): number of tokens threshold
    Returns:
        vocab(Vocabulary): vocabulary object
    """

    assert all([isinstance(arg, pd.DataFrame) for arg in args]), 'args must be pd.DataFrame'
    assert (isinstance(remove_punctuation, bool)), 'remove_punctuation must be bool data type'
    assert (isinstance(threshold, int)), 'threshold must be int data type'
    assert (threshold >= 0), 'threshold must be greater than or equals to zero'

    punctuations = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
                    ".", "?", "!", ",", ":", "-", "--", "...", ";"]

    counter = Counter()

    tokens = [self.startseq, self.endseq, self.unkseq, self.padseq]
    max_len = 300 # limit to 300

    for df in args:
        if all([col in df for col in ['tokens']]):
            for _, token in df['tokens'].iteritems():
                if remove_punctuation:
                    token = np.setdiff1d(token, punctuations).tolist()
                tokens.extend(token)
                counter.update(token)
                max_len = max(max_len, len(token) + 2)

    tokens = [token for token, cnt in counter.items() if cnt >= threshold]
    vocab = Vocabulary(tokens, max_len, self.unkseq)

    return vocab


class Vocabulary:

    def __init__(self, tokens, max_len, unkseq):

        self.vocab = sorted(list(set(tokens)))
        self.max_len = max_len

        self.word2idx = dict(map(reversed, enumerate(self.vocab)))
        self.idx2word = dict(enumerate(self.vocab))
        self.unkseq = unkseq

    def __call__(self, token):

        if not token in self.word2idx:
            return self.word2idx[self.unkseq]
        else:
            return self.word2idx[token]

    def __len__(self):
        return len(self.word2idx)