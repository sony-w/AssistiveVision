__author__ = 'sony-w'
__version__ = '1.0'

import argparse
import logging
import datetime
import os
import h5py
import boto3
import torch
import numpy as np

from loader.images import ImageS3

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs, fast_rcnn_inference_single_image

for name in ['boto', 'urllib3', 's3transfer', 'boto3', 'botocore', 'nose']:
    logging.getLogger(name).setLevel(logging.CRITICAL)
# set default log level
logging.basicConfig(
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


def list_keys(bucket_name, prefix, delimiter='/'):
    prefix = prefix[1:] if prefix.startswith(delimiter) else prefix
    bucket = boto3.resource('s3').Bucket(bucket_name)
    return (_.key for _ in bucket.objects.filter(Prefix=prefix).all())

def extract_features(predictor, img, num_objects=50):
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

            image = predictor.transform_gen.get_transform(img).apply_image(img)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = [{"image": image, "height": raw_height, "width": raw_width}]
            images = predictor.model.preprocess_image(inputs)

            # Run Backbone Res1-Res4
            features = predictor.model.backbone(images.tensor)

            # Generate proposals with RPN
            proposals, _ = predictor.model.proposal_generator(images, features, None)
            proposal = proposals[0]

            # Run RoI head for each proposal (RoI Pooling + Res5)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            features = [features[f] for f in predictor.model.roi_heads.in_features]
            box_features = predictor.model.roi_heads._shared_roi_transform(
                features, proposal_boxes
            )
            feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1

            # Predict classes and boxes for each proposal.
            pred_class_logits, pred_attr_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
            outputs = FastRCNNOutputs(
                predictor.model.roi_heads.box2box_transform,
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                predictor.model.roi_heads.smooth_l1_beta
            )

            probs = outputs.predict_probs()[0]
            boxes = outputs.predict_boxes()[0]

            attr_prob = pred_attr_logits[..., :-1].softmax(-1)
            max_attr_prob, max_attr_label = attr_prob.max(-1)

            # NMS
            for nms_thresh in np.arange(0.5, 1.0, 0.1):
                instances, ids = fast_rcnn_inference_single_image(
                    boxes, probs, image.shape[1:], 
                    score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=num_objects
                )
                if len(ids) == num_objects:
                    break

            instances = detector_postprocess(instances, raw_height, raw_width)
            roi_features = feature_pooled[ids].detach()
            max_attr_prob = max_attr_prob[ids].detach()
            max_attr_label = max_attr_label[ids].detach()
            instances.attr_scores = max_attr_prob
            instances.attr_classes = max_attr_label

            return instances, roi_features

def main(args):
    
    BUCKET_NAME = args.bucket_name
    NUM_OBJECTS = args.num_objects
    
    # get current date for log rotation
    date_str = datetime.date.today().isoformat()
    log_file = ''.join(['coco_features_', date_str, '.logs'])
    
    hdf5_path = 'features'
    os.makedirs(hdf5_path, exist_ok=True)
    
    # log handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(os.path.join(hdf5_path, log_file), mode='a')
    # log format
    c_format = logging.Formatter(fmt='%(name)s :: %(asctime)s :: %(levelname)s - %(message)s',
                                 datefmt='%b-%d-%y %H:%M:%S')
    f_format = logging.Formatter(fmt='%(name)s :: %(asctime)s :: %(levelname)s - %(message)s',
                                 datefmt='%b-%d-%y %H:%M:%S')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    # add log handlers to logger
    # logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    cfg = get_cfg()
    cfg.merge_from_file('bua/configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml')
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.MODEL.WEIGHTS = 'bua/weights/faster_rcnn_from_caffe_attr_original.pkl'  
    cfg.MODEL.DEVICE = device.type

    #cfg
    predictor = DefaultPredictor(cfg)

    image_s3 = ImageS3(bucket=BUCKET_NAME)
    
    hdf5_file = f'{hdf5_path}/coco_features.h5'
    mode = 'a'
    if os.path.exists(hdf5_file):
        mode = 'r+'

    with h5py.File(hdf5_file, mode) as hf:

        for dtype in ['train', 'val', 'test']:
            keys = list(list_keys('assistive-vision', prefix=f'coco/{dtype}'))[1:]

            for key in keys:

                feature_key = f'features_{dtype}_{key.split("/")[2]}'
                if not hf.get(feature_key):
                    _, features = extract_features(predictor, image_s3.getImageCV(key), num_objects=NUM_OBJECTS)

                    hf.create_dataset(feature_key, data=features)
                    logger.info(f'{feature_key} done..')
        
        logger.info('all complete..')
    
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Coco Features Extraction')
    parser.add_argument('--bucket_name', type=str, default='assistive-vision', help='S3 bucket name')
    parser.add_argument('--num_objects', type=int, default=50, help='max number of objects to detect')
    
    args = parser.parse_args()
    print(args)
    main(args)
