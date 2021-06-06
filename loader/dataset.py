__author__ = 'sony-w'
__version__ = '1.0'

import os
import torch
import pandas as pd

from .vizwiz import VizWiz
from .images import ImageS3

from itertools import repeat
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from tqdm import tqdm

class VizwizDataset(Dataset):
    """
        Vizwiz Dataset in torch tensor
    """
    
    def __init__(self, bucket='assistive-vision', vocab=None, dtype='train',
                startseq='<start>', endseq='<end>', unkseq='<unk>', padseq='<pad>',
                transformation=None,
                copy_img_to_mem=False,
                ret_type='tensor',
                device='cpu'):

        assert dtype in ['train', 'val', 'test'], "dtype value must be either 'train', 'val', or 'test'."
        assert ret_type in ['tensor', 'corpus', "return_type must be either 'tensor' or 'corpus'."]
        assert device in ['cpu', 'gpu'], "device must be either 'cpu' or 'gpu'."

        tqdm.pandas()

        self.imageS3 = ImageS3(bucket)
        self.dtype = dtype
        self.ret_type = ret_type
        self.copy_img_to_mem = copy_img_to_mem

        ann_path = os.path.join('annotations', ''.join([self.dtype, '.json']))
        vizwiz = VizWiz(annotation_file=ann_path)
        
        # load vizwiz to dataframe
        self.df = pd.DataFrame.from_dict(vizwiz.dataset['annotations'], orient='columns')
        images_df = pd.DataFrame.from_dict(vizwiz.dataset['images'], orient='columns')

        self.df = self.df.merge(images_df.rename({'id': 'image_id', 'text_detected': 'image_text_detected'}, axis=1), 
                                on='image_id', how='left')
        self.blob = {}
        self.loadImageAndTokenize()
    
    
    
    
    def loadImageAndTokenize(self):
        
        def _load_blob(path, f):
            _fpath = ''.join([path, '/', f])

            img = self.imageS3.getImage(_fpath)
            return img
    
        if self.copy_img_to_mem:
            
            fpath = os.path.join('vizwiz', self.dtype)
            
            cols = ['file_name']
            unique = self.df.groupby(by=cols, as_index=False).first()[cols]

            self.blob[unique['file_name']] = unique['file_name'].progress_apply(lambda f: _load_blob(fpath, f))
            
