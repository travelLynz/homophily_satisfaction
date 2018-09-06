import requests
import json
import settings as s
import os
import pandas as pd
import time

sight_keys = ['age', 'gender', 'emotions', 'ethnicity']

def get_sightcorp(img_dir):
    try:
        json_resp = requests.post( 'https://api-face.sightcorp.com/api/detect/',
              data  = { 'app_key' : s.SIGHTCORP_KEY, 'ethnicity' : True },
              files = { 'img'     : ( 'filename', open( img_dir, 'rb' ) ) } )
        people = json.loads(json_resp.text.replace('\n', ''))['people']
        num_faces = len(people) if people is not None else 0
        if num_faces != 1:
            print(str(num_faces) + " people detected.")
            return None
        else:
            return {key:people[0][key] for key in sight_keys}
    except Exception as e:
        print(e)
        return None

def create_sightcorp_table(img_dir, ids, cropped_dir=None, small_dir=None):
    people = []
    for i, n in zip(range(len(ids)), ids):
        if small_dir != None  and os.path.isfile(small_dir + str(n) + ".jpg"):
            img = small_dir + str(n) + ".jpg"
            res = None
        else :
            img = cropped_dir + str(n) + ".jpg" if (cropped_dir != None and os.path.isfile(cropped_dir + str(n) + ".jpg")) else img_dir + str(n) + ".jpg"
            res = get_sightcorp(img)

        if res is not None:
            people.append([os.path.basename(img).split('.')[0]]+[res[k] if k in res.keys() else None for k in sight_keys ])
        else:
            people.append([os.path.basename(img).split('.')[0]]+[None for k in sight_keys])
        time.sleep(1)
    return pd.DataFrame(people, columns=['name']+sight_keys)
