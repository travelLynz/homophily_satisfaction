from nltk.tokenize import sent_tokenize
import pandas as pd
import utils
def save_review_sentence(tbl,col,dir="", review_type='guest'):
    revset = {i:c for i, c in zip(tbl['id'], tbl[col])} if review_type=='guest' else {i:c for i, c in zip(tbl['idHostReview'], tbl[col])}
    din = dir +'so_cal/sent/in/'
    for i, c in revset.items():
        sents = sent_tokenize(c)

        for n, s in zip(range(len(sents)), sents):
            file = open(din + str(i) + '_' + str(n) + '.txt', 'w')
            file.write(s)
            file.close()

def get_sentence_level_sent(out_dir, table, review_type='guest'):
    df = pd.read_csv(out_dir)
    df[['so_cal_id', 'num_of_sent']] = df['File_Name'].map(lambda x: x.replace('.txt', '')).str.split('_', expand=True)
    dic= {}
    for i in df['so_cal_id'].unique():
        dic[i] = {}
    for i, r in df.iterrows():
        dic[r['so_cal_id']][r['num_of_sent']] = r['Score']
    return [utils.dic_to_list(dic[str(r['id'])]) for _, r in table.iterrows()] if review_type == 'guest' else [utils.dic_to_list(dic[str(r['idHostReview'])]) for _, r in table.iterrows()] 
    # for i in df['so_cal_id'].unique():
    #     dic_scores[i] = []
    #     for _,r in df[df['so_cal_id'] == i].sort_values(by='num_of_sent').iterrows():
    #         dic_scores[i].append(int(r['Score']))
    # return [dic_scores[str(r['id'])]for i, r in table.iterrows()]

def save_review_so_cal(tbl, col,dir="", review_type='guest'):
    revset = {i:c for i, c in zip(tbl['id'], tbl[col])} if review_type == 'guest' else {i:c for i, c in zip(tbl['idHostReview'], tbl[col])}
    din = dir + 'so_cal/overall/in/'
    for i, c in revset.items():
        file = open(din + str(i) + '.txt', 'w')
        file.write(c)
        file.close()

def get_overall_sent(out_dir, table, review_type='guest'):
    df = pd.read_csv(out_dir)
    df['so_cal_id'] = df['File_Name'].map(lambda x: int(x.replace('.txt', '')))
    table['so_cal'] = table.join(df.set_index('so_cal_id'), on='id')['Score'] if review_type=='guest' else table.join(df.set_index('so_cal_id'), on='idHostReview')['Score']
    return table
