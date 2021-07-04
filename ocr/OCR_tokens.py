from __future__ import print_function

__author__ = 'nima-m'
__version__ = '1.0'

import boto3
import cv2
import glob
import time
import json
import numpy as np
import os
import sys
from django.core.exceptions import ValidationError 

sys.path.append('/root')
from loader import images as im

client = boto3.client('sagemaker-runtime', region_name='us-east-1')

custom_attributes = "c000b4f9-df62-4c85-a0bf-7c525f9104a4"
endpoint_name = "mask-text-spotter-v3-endpoint"                                       # Your endpoint name.
content_type = "image/png"                                        # The MIME type of the input data in the request body.
accept = "application/json"                                              # The desired MIME type of the inference in the response.

# file_paths = sorted(glob.glob("test_images/*.jpeg"))
# print(len(file_paths))

#image_folder = 'test_imgs'
#abs_path = os.path.abspath(image_folder)
#regex_path = os.path.join(abs_path, '*.jpeg')
#file_paths = sorted(glob.glob(regex_path))
#print('Files : ', len(file_paths))

def get_ocr_tokens(img_name:str, bucket:str = 'assistive-vision', dataset:str='vizwiz', datatype:str='train') -> list:
    """Returns a list of the characters/words in an image. 
    image_name: takes the name of the image of the in S3"""
    
    img = np.array(im.ImageS3(bucket).getImage(f'{dataset}/{datatype}/{img_name}'))
    success, encoded_image = cv2.imencode('.jpg', img)
    payload = encoded_image.tobytes()

    t3 = time.time()
    try:
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType=content_type,
            Accept=accept,
            Body=payload
            )
    except:
        raise ValidationError("The Endpoint for processing OCR in images is currently turned off. Please ask Nima to turn on.")
    #print("FPS model prediction : ", 1/(time.time()-t3))
    else:
        output = json.loads(response["Body"].read().decode("utf-8"))
        result_words = output['words']
        return result_words
    #print("output : ", output.values())
    # boxes, classes, scores = output.values()

