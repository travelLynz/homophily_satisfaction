import os
import pandas as pd
import glob
import numpy as np
import utils
from nltk.tokenize import sent_tokenize

def save_review_sentence_short(revset, name='reviews', path="", split_size=100, review_type='guest'):
    for t, data in zip(range(split_size),np.array_split(revset, split_size)):
        din = os.path.join(path,'s140','sent', 'in', name  + str(t) + '.txt')
        file = open(din, 'w')
        ids = data.id if review_type == 'guest' else data.idHostReview
        for i, c in zip(ids, data.comments):
            sents = sent_tokenize(c)
            for n, sent in zip(range(len(sents)), sents):
                file.write(str(i) + "-" + str(n) + " : " + sent + "\r\n")
        file.close()

def get_sentence_level_sent(out_dir, table, review_type='guest'):
    df = pd.DataFrame()
    flist = glob.glob(out_dir + '*.txt')
    for filename in flist:
        fo = open(filename, "r")
        line = fo.readlines()
        df = df.append(line)
        fo.close()
    df = df.rename(columns={0:'string'})
    split_in = df.applymap(lambda x: x.split('","'))
    df['s140'], df['sid_string'] = split_in['string'].map(lambda x: int(x[0].replace('"',''))), split_in['string'].map(lambda x: x[1].split(':')[0].strip())
    df = df.drop(['string'], axis=1)
    df[['id','num_of_sent']]  = df.sid_string.str.split("-", expand=True)
    dic = {}
    for i in df['id'].unique():
        dic[i] = {}
    for i, r in df.iterrows():
        dic[r['id']][r['num_of_sent']] = r['s140']
    return [utils.dic_to_list(dic[str(r['id'])]) for i, r in table.iterrows()] if review_type=='guest' else [utils.dic_to_list(dic[str(r['idHostReview'])]) for i, r in table.iterrows()] 
    
def save_review_s140(revset, name, path="", review_type='guest'):
    for n, data in zip(range(40),np.array_split(revset, 40)):
        din = os.path.join(path, 's140', "overall","in", name + '_' + str(n) + '.txt')
        file = open(din, 'w')
        ids = data.id if review_type == 'guest' else data.idHostReview
        for i, c in zip(ids, data.comments):
            file.write(str(i) + " : " + c + "\r\n")
        file.close()

def read_s140_scores(out_dir, table, review_type='guest'):
    df = pd.DataFrame()
    flist = glob.glob(out_dir + '*.txt')
    for filename in flist:
        fo = open(filename, "r")
        line = fo.readlines()
        df = df.append(line)
        fo.close()
    df = df.rename(columns={0:'string'})
    print(len(df), df.columns)
    split_in = df.applymap(lambda x: x.split('","'))
    df['s140'], df['sid'] = split_in['string'].map(lambda x: int(x[0].replace('"',''))), split_in['string'].map(lambda x: x[1].split(':')[0].strip())
    df = df.drop(['string'], axis=1)
    df['sid'] = df['sid'].astype(int)
    #a = list(table.join(df.set_index('sid'), on='id')['s140'])
    table['s140'] = table.join(df.set_index('sid'), on='id')['s140'] if review_type == 'guest' else table.join(df.set_index('sid'), on='idHostReview')['s140']
    return table
