__author__ = 'sony-w'
__version__ = '1.0'

import boto3
import numpy as np
import logging

from PIL import Image
from io import BytesIO

from .bucket import BucketS3

from botocore.exceptions import ClientError

class ImageS3(BucketS3):
    """Utility class to read images from AWS S3 bucket"""
    
    def __init__(self, bucket='assistive-vision', aws_access_key_id=None, aws_secret_access_key=None, region_name=None):
        super(ImageS3, self).__init__(bucket=bucket, aws_access_key_id=aws_access_key_id, 
                                      aws_secret_access_key=aws_secret_access_key, region_name=region_name)
        
    
    def getImage(self, key_path):
        """
        Retrieve actual image
        
        Parameters:
            key(string): image file path in S3
        
        Returns:
            image: numpy array of image's pixel value with dimension of (height, width, depth)
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key_path)['Body'].read()
            image = Image.open(BytesIO(response))
            
            #image = np.asarray(bytearray(response.read()), dtype="uint8")
            #image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        except ClientError as e:
            logging.error(e)
            return None

        return image
    
    