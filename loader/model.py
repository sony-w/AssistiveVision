__author__ = 'sony-w'
__version__ = '1.0'

import torch
import os
import json
import pickle

from pathlib import Path
from io import BytesIO
from .bucket import BucketS3

from botocore.exceptions import ClientError

class ModelS3(BucketS3):
    """Utility class to save and load model to and from AWS S3 bucket"""
    
    def __init__(self, bucket='assistive-vision', aws_access_key_id=None, aws_secret_access_key=None, region_name=None, 
                 is_sagemaker=False, logger=None):
        
        super(ModelS3, self).__init__(bucket=bucket, aws_access_key_id=aws_access_key_id, 
                                      aws_secret_access_key=aws_secret_access_key, region_name=region_name, is_sagemaker=is_sagemaker, logger=logger)
        
    
    
    def save(self, state, local_path, key_path):
        """
        Save model state
        
        Parameters:
            state(dict): model state
            local_path(string): local file path to save the model state
            key_path(string): s3 file path to save the model state
        """
        try:
            path = Path(local_path)
            path.parent.mkdir(parents=True, exist_ok=True) 
            
            torch.save(state, local_path)
            self.s3_resource.Bucket(self.bucket).upload_file(local_path, key_path)
        except ClientError as e:
            self.logger.error(e)
    
    
    def load(self, local_path, key_path, overwrite=True):
        """
        Load model state
        
        Parameters:
            local_path(string): local file path
            key_path(string): s3 file path
            overwrite(boolean): overwrite local file if exists from s3 file when True
        
        Returns:
            image: numpy array of image's pixel value with dimension of (height, width, depth)
        """
        state = None
        try:
            if os.path.exists(local_path) and os.path.isfile(local_path):
                if overwrite:
                    self.s3_resource.Bucket(self.bucket).download_file(key_path, local_path)

            else:
                self.s3_resource.Bucket(self.bucket).download_file(key_path, local_path)
            
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            state = torch.load(local_path, map_location=device)
            
        except ClientError as e:
            self.logger.error(e)
        
        return state
    
    
    def save_captions(self, input_dict, key_path):
        """
        Save model's generated output captions
        
        Parameters:
            input_dict(dict): captions in vizwiz output format
            key_path(str): s3 file path
        """
        try:
            if self.is_sagemaker:
                json.dump(input_dict, key_path, indent=6, skipkeys=True)
            else:
                self.s3_client.put_object(Body=json.dumps(input_dict, indent=6, skipkeys=True), Bucket=self.bucket, Key=key_path)
            
        except ClientError as e:
            self.logger.error(e)
            
    
    def load_captions(self, key_path):
        
        json_loads = None
        try:
            if self.is_sagemaker:
                json_loads = json.load(key_path)
            else:
                response = self.s3_client.get_object(Bucket=self.bucket, Key=key_path)['Body'].read().decode('utf-8')
                json_loads = json.loads(response)
            
        except ClientError as e:
            self.logger.error(e)
        
        return json_loads
    
    
    def save_pkl(self, model, local_path, key_path):
        """
        Save model as pickle
        
        Parameters:
            model(model): model binary
            local_path(string): local file path to save the model state
            key_path(string): s3 file path to save the model state
        """
        try:
            path = Path(local_path)
            path.parent.mkdir(parents=True, exist_ok=True) 
            
            with open(local_path, 'wb') as file:
                pickle.dump(model, file)
            self.s3_resource.Bucket(self.bucket).upload_file(local_path, key_path)
        except ClientError as e:
            self.logger.error(e)
    
    
    def load_pkl(self, local_path, key_path, overwrite=True):
        """
        Load model pickle
        
        Parameters:
            local_path(string): local file path
            key_path(string): s3 file path
            overwrite(boolean): overwrite local file if exists from s3 file when True
        
        Returns:
            model: model binary
        """
        model = None
        try:
            if os.path.exists(local_path) and os.path.isfile(local_path):
                if overwrite:
                    self.s3_resource.Bucket(self.bucket).download_file(key_path, local_path)

            else:
                self.s3_resource.Bucket(self.bucket).download_file(key_path, local_path)
            
            with open(local_path, 'rb') as file:
                model = pickle.load(local_path)
            
        except ClientError as e:
            self.logger.error(e)
        
        return model