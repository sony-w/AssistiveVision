__author__ = 'sony-w'
__version__ = '1.0'

import boto3
import logging
import os

class BucketS3:
    
    def __init__(self, bucket, aws_access_key_id=None, aws_secret_access_key=None, region_name=None, is_sagemaker=False, logger=None):

        self.is_sagemaker = is_sagemaker
        
        if aws_access_key_id is not None and aws_secret_access_key is not None and region_name is not None:
            self.s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, 
                                          aws_secret_access_key=aws_secret_access_key, 
                                          region_name=region_name)

            self.s3_resource = boto3.resource('s3', aws_access_key_id=aws_access_key_id, 
                                              aws_secret_access_key=aws_secret_access_key, 
                                              region_name=region_name)

        else:
            self.s3_client = boto3.client('s3')
            self.s3_resource = boto3.resource('s3')

        self.bucket = bucket
        
        self.logger = logger
        if self.logger is None:
            self.logger = logging
        
        
