import urllib.request
from PIL import Image
from flask import request
import numpy as np
import json


def detect_masks(): 
 
    img = request.files['image']
    img = Image.open(img) 
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    
            
    # Request data goes here
    data = { "data": [img.tolist()]
    }
    
    
    body = str.encode(json.dumps(data))
    
    url = 'http://mask-detection-endpoint.centralindia.azurecontainer.io/score'
    api_key = 'UyLioeNN1nAggHxyBRs4z6VJbJ7pGdq5' # Replace this with the API key for the web service
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}
    

    req = urllib.request.Request(url, body, headers)

    try:    
        response = urllib.request.urlopen(req)
        result = response.read()
        y_hat = json.loads(result)
        prediction_classes = {0:' Correctly Masked Face', 1:'Incorrectly Masked Face', 2:'No Mask' }
        predictions = np.argmax(y_hat ,axis=1)
        return prediction_classes[predictions[0]]

    except urllib.error.HTTPError as error:
        return "The request failed with status code: " + str(error.code)
