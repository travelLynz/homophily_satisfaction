import os
import glob
import pandas as pd
import indicoio

indicoio.config.api_key = 'b170593a11c9ad0d814fc1e2f2d9a1f5'

def update_list(num_done, id_list):
    ids = pd.read_csv(id_list)
    done = ids[:num_done] #set([f.split("/")[-1].split(".")[0] for f in glob.glob(img_dir + '/*.jpg') if os.path.isfile(f)])
    rest = ids[num_done:]
    print("Saving Files")
    done.to_csv('done.csv', index=False)
    rest.to_csv('rest.csv', index=False)
    print("Done")

def get_num_of_people(img_dir, ids):
    num_of_people = []
    for i in ids:
        pwd = os.path.join(img_dir, str(i) +'.jpg')
        if os.path.isfile(pwd):
            num_of_people.append(len(indicoio.facial_localization(pwd)))
        else:
            num_of_people.append(None)
    return num_of_people
