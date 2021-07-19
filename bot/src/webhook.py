import boto3
import json
import os
import tweepy
import urllib

from captioner_msft import get_caption
from description_request import DescriptionRequest

def get_twitter_api():
    print("Get credentials")
    consumer_key = os.getenv("consumer_key")
    consumer_secret = os.getenv("consumer_secret")
    access_token = os.getenv("access_token")
    access_token_secret = os.getenv("access_token_secret")

    print("Authenticate")
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    return tweepy.API(auth)

def ok_msg(msg):
    return { 'statusCode': 200, 'message': msg }
    
def get_img_labels(client, img_bytes):
    response = client.detect_labels(
        Image={
            'Bytes': img_bytes,
        },
        MaxLabels=5,
        MinConfidence=80
    )
    return response
    
def get_img_text(client, img_bytes):
    response = client.detect_text(
        Image={
            'Bytes': img_bytes,
        },
        Filters={
            'WordFilter': {
                'MinConfidence': 80
            }
        }
    )
    return response
    
def post_reply(api, reply_to_id, msg, is_test = False):
    print(f'replying to {reply_to_id} with: {msg}')
    if len(msg) > 200:
        print(f'truncating {len(msg)} character message')
        msg = msg[:200] + '...'
    if is_test:
        print('not posting status since this is a test')
        return type('',(object,),{"id": 13132})()
    else:
        status = api.update_status(status = msg, in_reply_to_status_id = reply_to_id, auto_populate_reply_metadata=True)
        
    return status

def caption_to_description(cap):
    return f"I am just a bot, but I think it is: {cap}"
    
def labels_to_description(labels):
    print(labels)
    items = [f"I am {label['Confidence']:.2f} confident there is a {label['Name']}." for label in labels[:3]]
    joined_items = '\n'.join(items)
    return f"I am just a bot, but I've looked at it, and...\n{joined_items}"
    
def text_to_description(detections):
    print(detections)
    fragments = ['"' + text['DetectedText'] + '"' for text in detections if text['Type'] == "LINE"]
    joined_items = ' '.join(fragments)
    return f"I also see some text. It says: {joined_items}"

def handler(event, context):
    print(event)
    is_test = 'is_test' in event and event['is_test']
    hook_event = json.loads(event['body'])

    req = DescriptionRequest(hook_event)
    if not req.is_valid:
        print(req.message)
        return ok_msg(req.message)
    print(f'received request to describe tweet, req id: {req.tweet_id}, target id: {req.target_tweet_id}')
    
    api = get_twitter_api()
    img_status = api.get_status(req.target_tweet_id)
    
    if not 'media' in img_status.entities or len(img_status.entities['media']) != 1:
        print(f'no media or multiple media: {img_status.entities}')
        return ok_msg('We can only describe a single image.')
    img_url = img_status.entities['media'][0]['media_url_https']
    img_bytes = urllib.request.urlopen(img_url).read()
    
    rekog_cli = boto3.client('rekognition')
    labels = get_img_labels(rekog_cli, img_bytes)
    if not 'Labels' in labels or len(labels['Labels']) == 0:
        post_reply(api, req.tweet_id, "Sorry, but I couldn't identify anything in the image", is_test)
        return ok_msg('Description generation failed')

    cap = get_caption(img_bytes)
        
    description = caption_to_description(cap)
    label_tweet = post_reply(api, req.tweet_id, description, is_test)
    
    image_text = get_img_text(rekog_cli, img_bytes)
    if not 'TextDetections' in image_text or len(image_text['TextDetections']) == 0:
        print('no text found')
    else:
        text_desc = text_to_description(image_text['TextDetections'])
        text_tweet = post_reply(api, label_tweet.id, text_desc, is_test)
    
    return ok_msg('success')
