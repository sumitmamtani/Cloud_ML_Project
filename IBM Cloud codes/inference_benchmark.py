import numpy as np
from PIL import Image
import requests
import time

runs=[1, 4, 8, 16, 32, 64, 100, 128, 256, 512]
# runs=[1]
start = time.time()
image_url = "https://i.imgur.com/6qsCz2W.png"
url = 'http://linserv1.cims.nyu.edu:49101/predict'
for run in runs:
    for i in range(run):

        url = "http://linserv1.cims.nyu.edu:49101/predict"

        payload={}
        files=[
        ('inputFile',('two.jpeg',open('/Users/adityapandey/Downloads/mnist/two.jpeg','rb'),'image/jpeg'))
        ]
        headers = {
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Origin': 'null',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36'
        }

        response = requests.request("POST", url, headers=headers, data=payload, files=files)
    end = time.time()
    print(end - start)
    time.sleep(2)