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
import torchvision.datasets as datasets
import sys


sys.path.append('/root/coco/PythonAPI')
sys.path.append('/root/coco/PythonAPI/pycocotools')

from pycocotools.coco import COCO


class VizwizDataset(Dataset):
    """
        Vizwiz Dataset in torch tensor
    """
    
    def __init__(self, bucket='assistive-vision', vocabulary=None, dtype='train',
                startseq='<start>', endseq='<end>', unkseq='<unk>', padseq='<pad>',
                transformations=None,
                copy_img_to_mem=False,
                ret_type='tensor',
                ret_raw=False,
                device=torch.device('cpu'),
                partial=None,
                data = 'vizwiz'):

        assert dtype in ['train', 'val', 'test'], "dtype value must be either 'train', 'val', or 'test'."
        assert ret_type in ['tensor', 'corpus', "return_type must be either 'tensor' or 'corpus'."]

        self.imageS3 = ImageS3(bucket)
        self.dtype = dtype
        self.ret_type = ret_type
        self.ret_raw = ret_raw
        self.copy_img_to_mem = copy_img_to_mem
        
        self.device = device
        self.torch = torch.cuda if (self.device.type == 'cuda') else torch
        
        self.__getitem__fn = self.__getitem__corpus if ret_type == 'corpus' else self.__getitem__tensor
        self.data = data
        if self.data == 'vizwiz':
            ann_path = os.path.join('annotations' if self.data =='vizwiz' else 'annotations_coco', ''.join([self.dtype, '.json']))
            vizwiz = VizWiz(annotation_file=ann_path)
        
        else:
            ann_path = os.path.join('annotations_coco', ''.join([f"captions_{self.dtype}2017", '.json']))
            vizwiz = COCO(annotation_file=ann_path)
        
        # load vizwiz to dataframe
        self.df = pd.DataFrame.from_dict(vizwiz.dataset['annotations'], orient='columns')
        images_df = pd.DataFrame.from_dict(vizwiz.dataset['images'], orient='columns')

        
        if not self.df.empty:
            if self.data == 'vizwiz':
                self.df = self.df.merge(images_df.rename({'id': 'image_id', 'text_detected': 'image_text_detected'}, axis=1), 
                                    on='image_id', how='left')
            else:
                self.df = self.df.merge(images_df.rename({'id': 'image_id'}, axis=1), 
                                    on='image_id', how='left')
                
        else:
            self.df = images_df.rename({'id': 'image_id', 'text_detected': 'image_text_detected'}, axis=1)
        
        if partial is not None: 
            if isinstance(partial, int):
                if partial > 0:
                    self.df = self.df.iloc[:partial]
                else: raise ValueError('partial must be greater than zero.')
            else: raise TypeError('partial must be an int value.')
                

        # use multiprocessing Manager for parallelization
        self.blob = mp.Manager().dict()
        
        self.transformations = transformations if transformations is not None else transforms.Compose(
        [
            transforms.ToTensor()
        ])
        
        self.startseq = startseq.strip()
        self.endseq = endseq.strip()
        self.unkseq = unkseq.strip()
        self.padseq = padseq.strip()
        
        self.loadImageAndCorpus()
        
        if vocabulary is None:
            self.vocabulary = self.__construct_vocab(self.startseq, self.endseq, self.unkseq, self.padseq)
        else:
            self.vocabulary = vocabulary
    
    
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
            fpath = os.path.join('vizwiz' if self.data=='vizwiz' else 'coco', self.dtype)
            
            cols = ['file_name']
            unique = self.df.groupby(by=cols, as_index=False).first()[cols]
            
            # parallelize with multiprocessing Process
            procs = [mp.Process(target=_load_blob, args=(self.blob, fpath, f)) for _, f in unique['file_name'].iteritems()]
            [p.start() for p in procs]
            [p.join() for p in procs]
            
            print('done!!')
        
        #if self.ret_type == 'corpus':
        if 'caption' in self.df:
            print('tokenizing caption...', end=' ')
            # faster token counts for each caption with vectorization
            self.df[['tokens', 'tokens_count']] = pd.DataFrame(np.row_stack(
                np.vectorize(_token_counts, otypes=['O'])(self.df['caption'])), index=self.df.index)
        else:
            print('no caption found...', end=' ')
        
        print('done!!')
    
    def getVocab(self):
        return self.vocabulary
    
    @property
    def pad_value(self):
        return 0
    
    def __getitem__(self, idx: int):
        return self.__getitem__fn(idx)

    def __len__(self):
        return len(self.df)

    def __getitem__corpus(self, idx: int):
        """
        Retrieve image blob, tokens, and tokens count from given index in raw format
        
        Parameters:
            idx(int): index
        
        Returns:
            img(heigth, width, depth): image blob
            tokens(string): array of tokens if tokens and tokens count exist
            tokens_count(int): array of tokens length if tokens and tokens count exist
        """
        
        row = self.df.iloc[idx]
        
        fname = row['file_name']
        fpath = os.path.join('vizwiz' if self.data=='vizwiz' else 'coco', self.dtype, fname)
        img = self.transformations(
            self.blob.get(fname, self.imageS3.getImage(fpath))).to(self.device)
        
        if all([r in row for r in ['tokens', 'tokens_count']]):
            return img, row['tokens'], row['tokens_count'], fname
        
        return img, fname
    
    
    def __getitem__tensor(self, idx: int):
        """
        Retrieve image blob, tokens, and tokens count from given index in tensor format
        
        Parameters:
            idx(int): index
        
        Returns:
            img(heigth, width, depth): image blob
            tokens(string): array of tokens if tokens and tokens count exist
            tokens_count(int): array of tokens length if tokens and tokens count exist
        """
        
        row = self.df.iloc[idx]
        
        fname = row['file_name']
        fpath = os.path.join('vizwiz' if self.data=='vizwiz' else 'coco', self.dtype, fname)
        img_tensor = self.transformations(
            self.blob.get(fname, self.imageS3.getImage(fpath))).to(self.device)
        
        ## TODO: use better token embedding
        if all([r in row for r in ['tokens']]):
            tokens = [self.startseq] + row['tokens'] + [self.endseq]
        
            tokens_tensor = self.torch.LongTensor(self.vocabulary.max_len).fill_(self.pad_value)
            tokens_tensor[:len(tokens)] = self.torch.LongTensor([self.vocabulary(token) for token in tokens])
            
            return img_tensor, tokens_tensor, len(tokens), fname
        
        return img_tensor, fname
    
    
    def __construct_vocab(self, startseq, endseq, unkseq, padseq):
        """
        Generate vocabulary object from all the captions
        
        Returns:
            vocab(Vocabulary): vocabulary object
        """
        
        tokens = [] # [self.startseq, self.endseq, self.unkseq, self.padseq]
        max_len = 0
        
        for _, token in self.df['tokens'].iteritems():
            tokens.extend(token)
            max_len = max(max_len, len(token) + 2)
        
        vocab = Vocabulary(tokens, max_len, startseq, endseq, unkseq, padseq)
        
        return vocab 
    

class Vocabulary:
    
    def __init__(self, tokens, max_len, startseq='<start>', endseq='<end>', unkseq='<unk>', padseq='<pad>'):
        
        self.startseq = startseq.strip()
        self.endseq = endseq.strip()
        self.unkseq = unkseq.strip()
        self.padseq = padseq.strip()
        
        t = [self.startseq, self.endseq, self.unkseq, self.padseq]
        t.extend(tokens)
        
        self.vocab = sorted(list(set(t)))
        self.max_len = max_len
        
        self.word2idx = dict(map(reversed, enumerate(self.vocab)))
        self.idx2word = dict(enumerate(self.vocab))
        
    
    def __call__(self, token):
        
        if not token in self.word2idx:
            return self.word2idx[self.unkseq]
        else:
            return self.word2idx[token]
        
    def __len__(self):
        return len(self.word2idx)