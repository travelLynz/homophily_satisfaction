import pandas as pd
import numpy as np
import json
import settings as s
import re
import seaborn as sns
import analysis
import matplotlib.pyplot as plt

def plot_pie_graph(table, col, title=""):

    dist = pd.DataFrame(table[col].value_counts().reset_index())
    dist.columns = [col, 'count']
    labels = dist[col]
    counts = dist['count']

    # Data to plot
    #colors = ['lightgrey', 'orange', 'yellowgreen', 'lightcoral']
    #explode = (0.1, 0, 0, 0)  # explode 1st slice

    # Plot
    plt.pie(counts, labels=None,  shadow=True, startangle=140)
    plt.legend( loc = 'lower right', labels=['%s, %1.1f %%' % (l, s) for l, s in zip(labels, counts/np.sum(counts)*100)])

    plt.axis('equal')
    plt.title(title)
    plt.show()

    return dist

def max_len(table, column):
    return table.column.fillna('').map(len).max()

def get_comments(table, id_col, id, col):
    return "".join(table[table[id_col] == id][col])

def cut_list(cut_point, id_list):
    ids = pd.read_csv(id_list, index_col=0)
    done = ids[:cut_point] #set([f.split("/")[-1].split(".")[0] for f in glob.glob(img_dir + '/*.jpg') if os.path.isfile(f)])
    rest = ids[cut_point:]
    print("Saving Files")
    done.to_csv('done.csv')
    rest.to_csv('rest.csv')
    print("Done")

def trim_column_names(tb):
    return tb.rename(columns=lambda x: x.strip())

def clean_quotations(s):
    return str(s).replace('"','') if s not in [None, '', 'null'] else None

def save_dict_as_json(dict, name, dir='./'):

    js = json.dumps(dict)
    f = open(dir+name+'.json','w')
    f.write(js)
    f.close()

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

def create_value_counts_table(tbl, col, unique_col_name):
    result = tbl[col].value_counts(dropna=False).rename_axis(unique_col_name).reset_index(name='counts')
    result['%'] = result['counts'].map(lambda x: format(x*100/np.sum(result['counts']), '.2f'))
    return result

def plot_bar_graph(tb, xcol, ycol, xlabel, ylabel, title):
    ind = np.arange(len(tb))  # the x locations for the groups
    width = 0.4
    plt.bar(ind, tb[ycol],width, color="lightblue")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.xticks(ind, tb[xcol])
    plt.show()

def plot_dist_graph(tbl, col, ignore_nulls=False, title="", kde=True):
    vals = tbl[~tbl[col].isnull()][col] if ignore_nulls else tbl[col]
    sns.distplot(vals, kde=kde)
    plt.title(title)
    plt.show()

def get_decreased_percent(num, den):
    return 100-(len(num)*100/len(den))

def tokenize(input):
    my_re = re.compile(r"[\w']+|'m|'t|'ll|'ve|'d|'s|\'")
    return my_re.findall(input)

def print_summary(col, kde=True):
    sns.distplot(col, kde=kde)
    plt.show()
    sns.boxplot(col)
    plt.show()
    print("Average:", np.average(col))
    print("Min:", np.min(col))
    print("Max:", np.max(col))
    print("Variance:", np.var(col))
    print("Std deviation:", np.sqrt(np.var(col)))

def minmaxscale(vals):
    return (vals - np.min(vals))/(np.max(vals) - np.min(vals))

def minmaxscale_sent(tbl, col, isString=False):
    val_set = set([])
    if isString:
        tbl[col] = tbl[col].map(lambda x : np.array(x.replace('[', '').replace(']', '').split(", ")).astype(np.float) if type(x) == str else [x])
    for i in tbl[col]:
        val_set = val_set.union(set([float(v) for v in i]))
    min = np.min(list(val_set))
    max = np.max(list(val_set))
    vals = []
    for r in tbl[col]:
        row_vals = []
        for v in r:
            row_vals.append(format((float(v)-min)/(max - min), ".3f"))
        vals.append(row_vals)
    return vals


def to_float(a):
    return [float(i) for i in a ]

def clip_outliers(x, low, high):
    if x < low:
        return low
    elif x> high:
        return high
    else:
        return x

def clip_outliers_sent(x, low, high):
    vals = []
    for i in x:
        if float(i) < float(low):
            vals.append(low)
        elif float(i)> float(high):
            vals.append(high)
        else:
            vals.append(i)
    return vals

def get_new_minmax(vals, fence_type="inner"):
    q1 = np.percentile(vals, 25)
    q3 = np.percentile(vals, 75)
    iqr = q3 - q1
    margin = iqr*1.5 if fence_type == 'inner' else iqr*3
    min = q1 - margin
    max = q3 + margin
    return (min, max)

def get_new_minmax_sent(vals, fence_type="inner"):
    new_vals = [float(j) for i in vals for j in i]
    q1 = np.percentile(new_vals, 25)
    q3 = np.percentile(new_vals, 75)
    iqr = q3 - q1
    margin = iqr*1.5 if fence_type == 'inner' else iqr*3
    min = q1 - margin
    max = q3 + margin
    return (min, max)


def transform_tool(src_tbl, tool, trans_type= None, clip=None):
    src_tbl.is_copy = False
    current_tool = tool
    if trans_type != None:
        src_tbl[current_tool+"_transformed"] = transform_data(src_tbl[current_tool], trans_type, True)
        current_tool = current_tool+"_transformed"
    if clip != None:
        #src_tbl[current_tool + '_norm'] = minmaxscale(new_tbl[tool])
        min, max = get_new_minmax(src_tbl[current_tool], clip)
        src_tbl[current_tool] = [clip_outliers(i, min, max) for i in src_tbl[current_tool]]
        src_tbl[current_tool] = minmaxscale(src_tbl[current_tool])
    return src_tbl

def transform_tool_sent(src_tbl, tool, trans_type= None, clip=None):
    src_tbl.is_copy = False
    current_tool = tool
    if trans_type != None:
        src_tbl[current_tool+"_transformed"] = transform_sent(src_tbl, current_tool, trans_type)
        src_tbl[current_tool+"_transformed"] = minmaxscale_sent(src_tbl, current_tool)
        current_tool = current_tool+"_transformed"
    if clip != None:
        min, max = get_new_minmax_sent(src_tbl[current_tool], clip)
        src_tbl[current_tool ] = [clip_outliers_sent(i, min, max) for i in src_tbl[current_tool]]
        src_tbl[current_tool] = minmaxscale_sent(src_tbl, current_tool)
    return src_tbl

def transform_data(data, trans_type, normalise=False):
    vals = []
    if trans_type == 'exp':
        vals = 2**data
    elif trans_type == 'log':
        vals = np.log(data+1)
    elif trans_type == 'log10':
        vals = np.log10(data+1)
    elif trans_type == 'sqrt':
        vals = np.sqrt(data)
    elif trans_type == 'pwr':
        vals = data**2
    elif trans_type == 'multi':
        vals = data*2
    elif trans_type == 'arcsin':
        vals = np.arcsin(data)

    return minmaxscale(vals) if normalise else vals

def transform_score(src_tbl, tool):
    new_tbl = src_tbl[['id', 'comments', 'token_len', 'num_of_sents', tool]]
    new_tbl.is_copy = False

    if tool == 'afinn':
        a = src_tbl[tool]*(0.5/np.std(src_tbl[tool]))
        new_tbl[tool + '-ndist'] = np.log( a + np.abs(np.min(a)) + 1)
    elif tool == 'vader': #*(1/np.std(src_tbl[tool]))
        new_tbl[tool + '-ndist'] = 2**(src_tbl[tool])
    elif tool == 'opinion_freq':
        new_tbl[tool + '-ndist'] = np.log(src_tbl[tool] + np.abs(np.min(src_tbl[tool])) + 1)
    # elif tool == 'opinion_bow':
    #     new_tbl[tool + '-ndist'] = np.arcsin(src_tbl[tool])

    new_tbl[tool + '-norm'] = minmaxscale(new_tbl[tool])
    if tool not in ['s140', 'so_cal', 'opinion_bow'] :
        new_tbl[tool + '-ndist-norm'] = minmaxscale(new_tbl[tool + '-ndist'])
        new_tbl = analysis.split_into_levels(new_tbl, new_tbl, tool + '-ndist-norm')
    else:
        new_tbl = analysis.split_into_levels(new_tbl, new_tbl, tool + '-norm')

    return new_tbl

def transform_score_sent(src_tbl, tool):
    new_tbl = src_tbl[['id', 'comments', 'token_len', 'num_of_sents', tool]]
    new_tbl.is_copy = False
    vals = []
    if tool == 'afinn-sent':
        all = [j for i in src_tbl[tool] for j in eval(i)]
        min = np.min(np.array(all))
        for i, r in src_tbl.iterrows():
            vals.append(np.log10(eval(r[tool]) - min + 1))
        new_tbl[tool + '-ndist'] = vals
    elif tool == 'vader-sent':
        for i, r in src_tbl.iterrows():
            vals.append(np.arcsin(eval(r[tool])))
        new_tbl[tool + '-ndist'] = vals
    if tool == 'so_cal-sent':
        all = [j for i in src_tbl[tool] for j in eval(i)]
        min = np.min(np.array(all))
        for i, r in src_tbl.iterrows():
            vals.append(np.log(eval(r[tool]) - min + 1))
        new_tbl[tool + '-ndist'] = vals
    elif tool == 'opinion_bow-sent' or tool == 'opinion_freq-sent':
        all = [j for i in src_tbl[tool] for j in eval(i)]
        min = np.min(np.array(all))
        for i, r in src_tbl.iterrows():
            vals.append(np.log(eval(r[tool]) - min + 1))
        new_tbl[tool + '-ndist'] = vals
    # elif tool == 'vader': #*(1/np.std(src_tbl[tool]))
    #     new_tbl[tool + '-ndist'] = 2**(src_tbl[tool])
    # elif tool == 'opinion_freq':
    #     new_tbl[tool + '-ndist'] = np.log(src_tbl[tool] + np.abs(np.min(src_tbl[tool])) + 1)
    # elif tool == 'opinion_bow':
    #     new_tbl[tool + '-ndist'] = np.arcsin(src_tbl[tool])

    new_tbl[tool + '-norm'] = minmaxscale_sent(new_tbl, tool, True)
    if tool not in ['s140-sent']:
        new_tbl[tool + '-ndist-norm'] = minmaxscale_sent(new_tbl, tool + '-ndist')
    # if tool not in ['s140', 'so_cal', 'opinion_bow'] :
    #     new_tbl[tool + '-ndist-norm'] = minmaxscale(new_tbl[tool + '-ndist'])
    #     new_tbl = analysis.split_into_levels(new_tbl, new_tbl, tool + '-ndist-norm')
    # else:
    #     new_tbl = analysis.split_into_levels(new_tbl, new_tbl, tool + '-norm')

    return new_tbl

def transform_sent(tbl, col, transform="log"):
    vals = []
    if transform == 'exp':
        for i, r in tbl.iterrows():
            vals.append([2**float(v) for v in r[col]])
    elif transform == 'log':
        for i, r in tbl.iterrows():
            vals.append([np.log(float(v)+ 1) for v in r[col]])
    elif transform == 'log10':
        for i, r in tbl.iterrows():
            vals.append([np.log10(float(v)+1) for v in r[col]])
    elif transform == 'sqrt':
        for i, r in tbl.iterrows():
            vals.append([np.sqrt(float(v)) for v in r[col]])
    elif transform == 'pwr':
        for i, r in tbl.iterrows():
            vals.append([float(v)**2 for v in r[col]])
    elif transform == 'multi':
        for i, r in tbl.iterrows():
            vals.append([float(v)*2 for v in r[col]])
    elif transform == 'arcsin':
        for i, r in tbl.iterrows():
            vals.append([np.arcsin(float(v)) for v in r[col]])
    return vals

def normalize_sentence_scores(src_tbl, tool):
    new_tbl = src_tbl[['id', 'comments', tool]]
    new_tbl.is_copy = False

    new_tbl[tool + '-norm'] = minmaxscale_sent(new_tbl,tool, True)
    return new_tbl

def create_normalised_sent_table(src_tbl, tool, clip='outer'):
    new_tbl = src_tbl[['id', 'comments', tool]]
    new_tbl.is_copy = False

    new_tbl[tool + '_norm'] = minmaxscale_sent(new_tbl,tool, True)

    # if transform != None:
    #     new_tbl['trans_'+tool] = transform_sent(new_tbl, tool + '-norm', transform)
    #     new_tbl['trans_'+tool+'_norm'] = minmaxscale_sent(new_tbl, 'trans_' + tool)
    #     tool = 'trans_'+tool

    min, max = get_new_minmax_sent(new_tbl[tool], clip)

    new_tbl[tool + '_clipped'] = [clip_outliers_sent(i, min, max) for i in new_tbl[tool]]
    new_tbl[tool + '_clipped_norm'] = minmaxscale_sent(new_tbl, tool + '_clipped')
    return new_tbl

def dic_to_list(dic):
    return [dic[str(i)] for i in range(len(dic))]
def str_to_list(x):
    return np.array(x.replace('\'', '').replace('[', '').replace(']', '').split(", ")).astype(np.float)

def flatten(lists, convert=True):
    return [float(v) for l in lists for v in eval(l)] if convert else [v for l in lists for v in l]

def hasNumbers(x):
    return bool(re.search(r'\d', x))

def invert_dict(d):
    return dict([ (v, k) for k, v in d.items( ) ])

def convert_to_str(ls):
    return [str(i) if i != None else None for i in ls]

def convert_to_int(s):
    return {int(i ) for i in s}

def reduce_reviews(reviews, hosts, guests):
    host_ids = hosts.id.unique()
    guest_ids = guests.id.unique()

    reviews = reviews[(reviews.reviewer_id.isin(guest_ids)) & (reviews.recipient_id.isin(host_ids))]
    new_hosts, new_guests = (reviews.recipient_id.unique(), reviews.reviewer_id.unique())
    print("Total Reviews: %d \nUnique Hosts: %d \nUnique Guests: %d" % (len(reviews) , len(new_hosts), len(new_guests)))
    return (reviews , new_hosts, new_guests)

def hasNumbers(inputString):
     return any(char.isdigit() for char in inputString)
