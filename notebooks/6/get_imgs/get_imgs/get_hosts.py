import pandas as pd
import sys
import os
import glob
import numpy as np
import images as im

hosts_left = pd.read_csv('hosts_left.csv')

for n, data in zip(range(20),np.array_split(hosts_left, 20)):
        din = 'hosts_left_done_' + str(n) + '.csv'
        data['num_of_people'] = im.get_num_of_people('host_imgs', data['id'])
        data.to_csv(din)

