import os
import glob
import pandas as pd
import indicoio
import numpy as np
from PIL import Image
from prettytable import PrettyTable

indicoio.config.api_key = '335ab11a342bcba3624bb4c5ec108246'

def update_list(num_done, id_list):
    ids = pd.read_csv(id_list)
    done = ids[:num_done] #set([f.split("/")[-1].split(".")[0] for f in glob.glob(img_dir + '/*.jpg') if os.path.isfile(f)])
    rest = ids[num_done:]
    print("Saving Files")
    done.to_csv('done.csv', index=False)
    rest.to_csv('rest.csv', index=False)
    print("Done")

def get_num_of_people(img_dir, no_img_dir, ids):
    num_of_people = []
    for i in ids:
        pwd = os.path.join(img_dir, str(i) +'.jpg')
        no_pwd = os.path.join(no_img_dir, str(i) +'.jpg')
        if os.path.isfile(no_pwd):
            num_of_people.append(0)
        elif os.path.isfile(pwd):
            num_of_people.append(len(indicoio.facial_localization(pwd)))
        else:
            num_of_people.append(None)
    return num_of_people

def get_image_coordinates(img_dir, no_img_dir, ids):
    pic_ids = []
    box_coords = []
    for i in ids:
        pwd = os.path.join(img_dir, str(i) +'.jpg')
        no_pwd = os.path.join(no_img_dir, str(i) +'.jpg')
        if os.path.isfile(no_pwd):
            pic_ids.append(i)
            box_coords.append((None, None, None, None))
        elif os.path.isfile(pwd):
            faces = indicoio.facial_localization(pwd)
            if len(faces) == 0 :
                pic_ids.append(i)
                box_coords.append((None, None, None, None))
            else :
                for face in faces:
                    pic_ids.append(i)
                    box_coords.append((face['top_left_corner'][0], face['top_left_corner'][1], face['bottom_right_corner'][0], face['bottom_right_corner'][1]))
        else:
            pass
    return (pic_ids, box_coords )

def crop_images(img_dir, dest_dir, crop_table):
    for i, r in crop_table.iterrows():
        img = Image.open(img_dir + r['id'] + ".jpg")
        cropped_img = img.crop(eval(r['bounding_box']))
        cropped_img.save(dest_dir + r['id'] + '.jpg')

def print_pip_proportions(tbl, col):

    t = PrettyTable()

    t.add_column('total',[len(tbl)])

    nan = len(tbl[tbl[col].isnull()])
    t.add_column('n/a', [str(nan) + "(" + format(nan*100/len(tbl), ".2f") +"%)"])

    for i in range(5):
        res = len(tbl[tbl[col] == i])
        t.add_column(str(i), [str(res) + "(" + format(res*100/len(tbl), ".2f") +"%)"])
    others = len(tbl[tbl[col] >= 5])
    t.add_column('>=5', [str(others) + "(" + format(others*100/len(tbl), ".2f") +"%)"])
    print(t)

def clean_cognitive_table(cog_tbl):
    cog_tbl.noise = cog_tbl.noise.map(lambda x : x if x not in [np.nan ] else None)
    nlevels = []
    nvals = []
    ageQ = []
    for _, r in cog_tbl.iterrows():
        noise = r['noise']
        if noise == None:
            nlevels.append(None)
            nvals.append(np.nan)
            ageQ.append(np.nan)
        else:
            nlevels.append(eval(noise)['noiseLevel'])
            nvals.append(eval(noise)['value'])
            age = r['age']
            if age < 27:
                ageQ.append(1)
            elif age >= 27 and age < 33:
                ageQ.append(2)
            elif age >= 33 and age < 39:
                ageQ.append(3)
            else:
                ageQ.append(4)
    cog_tbl['noiseLevel'], cog_tbl['noiseValue'], cog_tbl['ageQ']  = nlevels, nvals, ageQ
    cog_tbl = cog_tbl.drop(["noise"], axis=1)
    return cog_tbl
