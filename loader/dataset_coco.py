__author__ = 'nima-m'
__version__ = '1.0'

import os
import torch
import pandas as pd
import numpy as np
import multiprocessing as mp
import nltk
import logging

from .images import ImageS3

from torch.utils.data import Dataset
from torchvision import transforms

from pycocotools.coco import COCO

from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs, fast_rcnn_inference_single_image


class COCODataset(Dataset):
    """
        COCO Dataset in torch tensor
    """
    
    def __init__(self, bucket='assistive-vision', vocabulary=None, dtype='train',
                 startseq='<start>', endseq='<end>', unkseq='<unk>', padseq='<pad>',
                 transformations=None,
                 predictor=None,
                 num_objects=10,
                 copy_img_to_mem=False,
                 ret_type='tensor',
                 ret_raw=False,
                 device=torch.device('cpu'),
                 partial=None,
                 aws_access_key_id=None, 
                 aws_secret_access_key=None, 
                 region_name=None,
                 is_sagemaker=False,
                 logger=None):

        assert dtype in ['train', 'val', 'test'], "dtype value must be either 'train', 'val', or 'test'."
        assert ret_type in ['tensor', 'corpus', "return_type must be either 'tensor' or 'corpus'."]

        self.imageS3 = ImageS3(bucket, aws_access_key_id=aws_access_key_id, 
                               aws_secret_access_key=aws_secret_access_key, region_name=region_name, is_sagemaker=is_sagemaker, logger=logger)
        self.dtype = dtype
        self.ret_type = ret_type
        self.ret_raw = ret_raw
        self.copy_img_to_mem = copy_img_to_mem
        self.is_sagemaker = is_sagemaker
        self.logger = logger
        if self.logger is None:
            self.logger = logging
        
        self.device = device
        self.torch = torch.cuda if (self.device.type == 'cuda') else torch
        
        self.__getitem__fn = self.__getitem__corpus if ret_type == 'corpus' else self.__getitem__tensor

        if self.dtype == 'test':
            ann_path = os.path.join('annotations_coco', ''.join([f"image_info_{self.dtype}2017", '.json']))
        else:
            ann_path = os.path.join('annotations_coco', ''.join([f"captions_{self.dtype}2017", '.json']))
        coco_ds = COCO(annotation_file=ann_path)
        
        self.df = pd.DataFrame()
        # load coco to dataframe
        if self.dtype != 'test':
            self.df = pd.DataFrame.from_dict(coco_ds.dataset['annotations'], orient='columns')
        images_df = pd.DataFrame.from_dict(coco_ds.dataset['images'], orient='columns')

        if not self.df.empty:
            self.df = self.df.merge(images_df.rename({'id': 'image_id'}, axis=1), 
                                    on='image_id', how='left')
        else:
            self.df = images_df.rename({'id': 'image_id'}, axis=1)
        
        if partial is not None: 
            if isinstance(partial, int):
                if partial > 0:
                    self.df = self.df.iloc[:partial]
                else: raise ValueError('partial must be greater than zero.')
            else: raise TypeError('partial must be an int value.')
                

        # use multiprocessing Manager for parallelization
        self.blob = mp.Manager().dict()
        
        self.predictor = predictor
        self.num_objects = num_objects
        self.transformations = transformations if transformations is not None else transforms.Compose(
        [
            transforms.ToTensor()
        ])
        
        self.startseq = startseq.strip()
        self.endseq = endseq.strip()
        self.unkseq = unkseq.strip()
        self.padseq = padseq.strip()
        
        self.loadImageAndCorpus()
        
        if vocabulary is not None:
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

            fpath = os.path.join('coco', self.dtype)
            if self.is_sagemaker:
                fpath = os.path.join('/opt/ml/input/data', self.dtype)
            
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

    def __extract_features(self, img):
        """
        Extract features from object detection
        Parameters:
            img(opencv): opencv type image
        Returns:
            instances(tensor): object detection predictions
            features(tensor): object detection features 
        """
        with torch.no_grad():
            raw_height, raw_width = img.shape[:2]

            image = self.predictor.transform_gen.get_transform(img).apply_image(img)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = [{"image": image, "height": raw_height, "width": raw_width}]
            images = self.predictor.model.preprocess_image(inputs)

            # Run Backbone Res1-Res4
            features = self.predictor.model.backbone(images.tensor)

            # Generate proposals with RPN
            proposals, _ = self.predictor.model.proposal_generator(images, features, None)
            proposal = proposals[0]

            # Run RoI head for each proposal (RoI Pooling + Res5)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            features = [features[f] for f in self.predictor.model.roi_heads.in_features]
            box_features = self.predictor.model.roi_heads._shared_roi_transform(
                features, proposal_boxes
            )
            feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1

            # Predict classes and boxes for each proposal.
            pred_class_logits, pred_attr_logits, pred_proposal_deltas = self.predictor.model.roi_heads.box_predictor(feature_pooled)
            outputs = FastRCNNOutputs(
                self.predictor.model.roi_heads.box2box_transform,
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                self.predictor.model.roi_heads.smooth_l1_beta
            )

            probs = outputs.predict_probs()[0]
            boxes = outputs.predict_boxes()[0]

            attr_prob = pred_attr_logits[..., :-1].softmax(-1)
            max_attr_prob, max_attr_label = attr_prob.max(-1)

            # NMS
            for nms_thresh in np.arange(0.5, 1.0, 0.1):
                instances, ids = fast_rcnn_inference_single_image(
                    boxes, probs, image.shape[1:], 
                    score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=self.num_objects
                )
                if len(ids) == self.num_objects:
                    break

            instances = detector_postprocess(instances, raw_height, raw_width)
            roi_features = feature_pooled[ids].detach()
            max_attr_prob = max_attr_prob[ids].detach()
            max_attr_label = max_attr_label[ids].detach()
            instances.attr_scores = max_attr_prob
            instances.attr_classes = max_attr_label

            return instances.to(self.device), roi_features.to(self.device)
    
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
        
        image_id = row['image_id']
        fname = row['file_name']
        
        fpath = os.path.join('coco', self.dtype, fname)
        if self.is_sagemaker:
            fpath = os.path.join('/opt/ml/input/data', self.dtype, fname)
        
        img = None
        if self.predictor:
            _, img = self.__extract_features(
                self.blob.get(fname, self.imageS3.getImageCV(fpath)))
        else:
            img = self.transformations(
                self.blob.get(fname, self.imageS3.getImage(fpath))).to(self.device)
        
        if all([r in row for r in ['tokens', 'tokens_count']]):
            return img, row['tokens'], row['tokens_count'], fname, image_id
        
        return img, fname, image_id

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

        image_id = row['image_id']
        fname = row['file_name']

        fpath = os.path.join('coco', self.dtype, fname)
        if self.is_sagemaker:
            fpath = os.path.join('/opt/ml/input/data', self.dtype, fname)

        img_tensor = None
        if self.predictor:
            _, img_tensor = self.__extract_features(
                self.blob.get(fname, self.imageS3.getImageCV(fpath)))
        else:
            img_tensor = self.transformations(
                self.blob.get(fname, self.imageS3.getImage(fpath))).to(self.device)
        
        ## TODO: use better token embedding
        if all([r in row for r in ['tokens']]):
            
            tokens = [self.startseq] + row['tokens'] + [self.endseq]
            if len(tokens) > self.vocabulary.max_len:
                _tokens = row['tokens'][:(self.vocabulary.max_len - 2)]
                tokens = [self.startseq] + _tokens + [self.endseq]
            
            tokens_tensor = self.torch.LongTensor(self.vocabulary.max_len).fill_(self.pad_value)

            try:
                tokens_tensor[:len(tokens)] = self.torch.LongTensor([self.vocabulary(token) for token in tokens])
            
            except RuntimeError as e:
                self.logger.error(e)
                self.logger.error(f'{image_id} :: {fname} :: vocab_max_len {self.vocabulary.max_len} :: tokens_len {len(tokens)}')
            
            return img_tensor, tokens_tensor, len(tokens), fname, image_id
        
        return img_tensor, fname, image_id
