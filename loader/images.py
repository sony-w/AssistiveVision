__author__ = 'sony-w'
__version__ = '1.0'

import boto3
import numpy as np
import cv2
import logging

from botocore.exceptions import ClientError

class ImageS3:
    """Utility class to read images from AWS S3 bucket"""
    
    def __init__(self, bucket, aws_access_key_id=None, aws_secret_access_key=None, region_name=None):
        
        if aws_access_key_id is not None and aws_secret_access_key is not None and region_name is not None:
            self.s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id,  
                                   aws_secret_access_key=aws_secret_access_key, 
                                   region_name=region_name)
        else:
            self.s3 = boto3.client('s3')

        self.bucket = bucket
            
    
    def getImage(self, key):
        """
        Retrieve actual image
        
        Parameters:
            key(string): image file path in S3
        
        Returns:
            image: numpy array of image's pixel value with dimension of (height, width, depth)
        """
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)['Body']

            image = np.asarray(bytearray(response.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        except ClientError as e:
            logging.error(e)
            return None

        return image
    
    