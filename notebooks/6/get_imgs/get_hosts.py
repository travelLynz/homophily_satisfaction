import pandas as pd
import sys
import os
import glob
import numpy as np
import images as im

hosts_left = pd.read_csv('hosts_left.csv')

hosts_left['num_of_people'] = im.get_num_of_people('host_imgs', hosts_left['id'])

hosts_left.to_csv("hosts_left_done.csv")
