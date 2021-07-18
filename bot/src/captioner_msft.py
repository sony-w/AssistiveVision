import json
import os
import requests

# place your Azure COGS CV subscription API Key and Endpoint below
API_KEY = os.getenv("msft_api_key")
ENDPOINT = os.getenv("msft_endpoint")

ANALYZE_URL = ENDPOINT + "/vision/v3.1/analyze"

def get_caption(img_bytes):

    """
    Function to get image caption when path to image file is given.
    Note: API_KEY and ANALYZE_URL need to be defined before calling this function.

    Parameters
    ----------
    path_to_image   : path of image file to be analyzed

    Output
    -------
    image caption
    """

    headers  = {
        'Ocp-Apim-Subscription-Key': API_KEY,
        'Content-Type': 'application/octet-stream'
    }
    params   = {
        'visualFeatures': 'Description',
        'language': 'en',
    }

    # get image caption using requests
    response = requests.post(ANALYZE_URL, headers=headers, params=params, data=img_bytes)
    results = json.loads(response.content)

    # return the first caption's text in results description
    caption = results['description']['captions'][0]['text']
    print(f'msft caption: {caption}')

    return caption