import sys
sys.path.append("../Cognitive-Face-Python/")
import cognitive_face as CF
import settings as s
import swagger_client
import time
import urllib3
import pandas as pd
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from swagger_client.rest import ApiException

CF.Key.set(s.COGNITIVE_KEY)
BASE_URL = 'https://uksouth.api.cognitive.microsoft.com/face/v1.0'  # Replace with your regional Base URL
CF.BaseUrl.set(BASE_URL)
cognitive_keys = ['age','gender','smile','noise']

def get_cognitive(img_dir): # create an instance of the API class
    api_instance = swagger_client.MediaApi()
    try:
        resp = CF.face.detect(img_dir, attributes=[','.join(cognitive_keys)])
        num_faces = len(resp) if resp is not None else 0
        if num_faces != 1:
            return None
        else:
            return resp[0]['faceAttributes']
    except Exception as e:
        print(e)
        time.sleep(60)
        get_cognitive(img_dir)
        return None

def create_cognitive_table(dir, ids):
    people = []
    for i, id in zip(range(len(ids)), ids):
        img = dir + str(id) + ".jpg"
        res = get_cognitive(img)
        if res is not None:
            people.append([id]+[res[k] if k in res.keys() else None for k in cognitive_keys])
        else:
            people.append([id]+[None for k in cognitive_keys])
        if (i+1) % 10 == 0:
            time.sleep(60)
    return pd.DataFrame(people, columns=['name']+cognitive_keys)
