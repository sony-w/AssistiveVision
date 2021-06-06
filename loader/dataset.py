__author__ = 'sony-w'
__version__ = '1.0'

import os
import torch
import pandas as pd
import numpy as np
import multiprocessing as mp
import nltk

from .vizwiz import VizWiz
from .images import ImageS3

from torch.utils.data import Dataset
from torchvision import transforms


class VizwizDataset(Dataset):
    """
        Vizwiz Dataset in torch tensor
    """
    
    def __init__(self, bucket='assistive-vision', vocab=None, dtype='train',
                startseq='<start>', endseq='<end>', unkseq='<unk>', padseq='<pad>',
                transformation=None,
                copy_img_to_mem=False,
                ret_type='tensor',
                ret_raw=False,
                device='cpu'):

        assert dtype in ['train', 'val', 'test'], "dtype value must be either 'train', 'val', or 'test'."
        assert ret_type in ['tensor', 'corpus', "return_type must be either 'tensor' or 'corpus'."]
        assert device in ['cpu', 'gpu'], "device must be either 'cpu' or 'gpu'."

        self.imageS3 = ImageS3(bucket)
        self.dtype = dtype
        self.ret_type = ret_type
        self.ret_raw = ret_raw
        self.copy_img_to_mem = copy_img_to_mem
        
        self.device = torch.device(device)
        self.torch = torch.cuda if (self.device.type == 'cuda') else torch
        
        self.__get_item__fn = self.__get_item__corpus if ret_type == 'corpus' else self.__get_item__tensor

        ann_path = os.path.join('annotations', ''.join([self.dtype, '.json']))
        vizwiz = VizWiz(annotation_file=ann_path)
        
        # load vizwiz to dataframe
        self.df = pd.DataFrame.from_dict(vizwiz.dataset['annotations'], orient='columns')
        images_df = pd.DataFrame.from_dict(vizwiz.dataset['images'], orient='columns')

        self.df = self.df.merge(images_df.rename({'id': 'image_id', 'text_detected': 'image_text_detected'}, axis=1), 
                                on='image_id', how='left')

        # use multiprocessing Manager for parallelization
        self.blob = mp.Manager().dict()
        
        self.transformation = transformation if transformation is not None else transforms.Compose(
        [
            transforms.ToTensor()
        ])
        
        self.startseq = startseq.strip()
        self.endseq = endseq.strip()
        self.unkseq = unkseq.strip()
        self.padseq = padseq.strip()
        
        self.loadImageAndCorpus()
        
        if vocab is None:
            self.vocab, self.word2idx, self.idx2word, self.max_len = self.__construct_vocab()
        else:
            self.vocab, self.word2idx, self.idx2word, self.max_len = vocab
    
    
    def loadImageAndCorpus(self):
        """
        Load images to memory and tokenize image captions
        """
        
        def _load_blob(blob, path, f):
            _fpath = ''.join([path, '/', f])
            blob[f] = self.imageS3.getImage(_fpath)

        ## TODO: use better tokenizer
        def _token_counts(s):
            tokens = nltk.word_tokenize(s.lower())
            return np.array([tokens, len(tokens)], dtype='object')

        
        if self.copy_img_to_mem:
            print('loading images to memory...', end=' ')
            fpath = os.path.join('vizwiz', self.dtype)
            
            cols = ['file_name']
            unique = self.df.groupby(by=cols, as_index=False).first()[cols]
            
            # parallelize with multiprocessing Process
            procs = [mp.Process(target=_load_blob, args=(self.blob, fpath, f)) for _, f in unique['file_name'].iteritems()]
            [p.start() for p in procs]
            [p.join() for p in procs]
            
            print('done!!')
        
        #if self.ret_type == 'corpus':
        print('tokenizing caption...', end=' ')
        # faster token counts for each caption with vectorization
        self.df[['tokens', 'tokens_count']] = pd.DataFrame(np.row_stack(
            np.vectorize(_token_counts, otypes=['O'])(self.df['caption'])), index=self.df.index)
        
        print('done!!')
    
    def getVocab(self):
        return self.vocab, self.word2idx, self.idx2word, self.max_len
    
    @property
    def pad_value(self):
        return 0
    
    def __get_item__(self, idx: int):
        return self.__get_item__fn(idx)

    def __len__(self):
        return len(self.df)

    def __get_item__corpus(self, idx: int):
        """
        Retrieve image blob, tokens, and tokens count from given index in raw format
        
        Parameters:
            idx(int): index
        
        Returns:
            img(heigth, width, depth): image blob
            tokens(string): array of tokens
            tokens_count(int): array of tokens length
        """
        
        row = self.df.iloc[idx]
        
        fname = row['file_name']
        fpath = os.path.join('vizwiz', self.dtype, fname)
        img = self.blob.get(fname, self.imageS3.getImage(fpath))
        
        return img, row['tokens'], row['tokens_count']
    
    
    def __get_item__tensor(self, idx: int):
        """
        Retrieve image blob, tokens, and tokens count from given index in tensor format
        
        Parameters:
            idx(int): index
        
        Returns:
            img(heigth, width, depth): image blob
            tokens(string): array of tokens
            tokens_count(int): array of tokens length
        """
        
        row = self.df.iloc[idx]
        
        tokens = [self.startseq + row['tokens'] + self.endseq]
        
        fname = row['file_name']
        fpath = os.path.join('vizwiz', self.dtype, fname)
        img_tensor = self.transformations(
            self.blob.get(fname, self.imageS3.getImage(fpath))).to(self.device)
        
        ## TODO: use better token embedding
        tokens_tensor = self.torch.LongTensor(self.max_len).fill_(self.pad_value)
        tokens_tensor[:len(tokens)] = self.torch.LongTensor([self.word2idx[token] for token in tokens])
        
        return img_tensor, tokens_tensor, len(tokens)
    
    
    def __construct_vocab(self):
        """
        Generate vocabs from all the captions
        
        Returns:
            vocab(string): sorted set of vocabs
            word2idx(dict): word to index dictionary
            idx2word(dict): index to word dictionary
            max_len(int): caption max length
        """
        
        tokens = [self.startseq, self.endseq, self.unkseq, self.padseq]
        max_len = 0
        
        for _, token in self.df['tokens'].iteritems():
            tokens.extend(token)
            max_len = max(max_len, len(token) + 2)
        
        vocab = sorted(list(set(tokens)))
        
        word2idx = dict(map(reversed, enumerate(vocab)))
        idx2word = dict(enumerate(vocab))
        
        return vocab, word2idx, idx2word, max_len
    
    