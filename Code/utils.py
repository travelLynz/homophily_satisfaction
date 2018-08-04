import pandas as pd
import numpy as np
import json
import settings as s
import re
import seaborn as sns
import matplotlib.pyplot as plt

def plot_pie_graph(table, col):

    dist = pd.DataFrame(table[col].value_counts().reset_index())
    dist.columns = [col, 'count']
    labels = dist[col]
    counts = dist['count']

    # Data to plot
    #colors = ['lightgrey', 'orange', 'yellowgreen', 'lightcoral']
    #explode = (0.1, 0, 0, 0)  # explode 1st slice

    # Plot
    plt.pie(counts, labels=labels,  shadow=True, startangle=140)
    plt.legend( loc = 'lower right', labels=['%s, %1.1f %%' % (l, s) for l, s in zip(labels, counts/np.sum(counts)*100)])

    plt.axis('equal')
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
        val_set = val_set.union(set(i))

    min = np.min(list(val_set))
    max = np.max(list(val_set))
    vals = []
    for r in tbl[col]:
        row_vals = []
        for v in r:
            row_vals.append(format((v-min)/(max - min), ".3f"))
        vals.append(row_vals)
    return vals


# def minmaxscale_sent(tbl, col, isString=False):
#     val_set = set([])
#     if isString:
#         for i in tbl[col]:
#             val_set = val_set.union(i.replace('[', '').replace(']', '').split(", "))
#         min = np.min([float(i) for i in val_set])
#         max = np.max([float(i) for i in val_set])
#         print(val_set)
#     else:
#         for i in tbl[col]:
#             val_set = val_set.union(set(i))
#         min = np.min(list(val_set))
#         max = np.max(list(val_set))
#     vals = []
#     for r in tbl[col]:
#         row_vals = []
#         i
#         for v in r:
#             row_vals.append(format((float(v)-min)/(max - min), ".3f"))
#         vals.append(row_vals)
#     return vals

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
        if i < low:
            vals.append(low)
        elif i> high:
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
    new_vals = [j for i in vals for j in i]
    q1 = np.percentile(new_vals, 25)
    q3 = np.percentile(new_vals, 75)
    iqr = q3 - q1
    margin = iqr*1.5 if fence_type == 'inner' else iqr*3
    min = q1 - margin
    max = q3 + margin
    return (min, max)

def create_normalised_table(src_tbl, tool, transform=None):
    new_tbl = src_tbl[['id', 'comments', tool]]
    new_tbl.is_copy = False

    new_tbl[tool + '-norm'] = minmaxscale(new_tbl[tool])

    if transform != None:
        if transform == 'exp':
            new_tbl['trans_'+tool] = 2**new_tbl[tool + '-norm']
        elif transform == 'log':
            new_tbl['trans_'+tool] = np.log(new_tbl[tool + '-norm']+1)
        elif transform == 'log10':
            new_tbl['trans_'+tool] = np.log10(new_tbl[tool + '-norm']+1)
        elif transform == 'sqrt':
            new_tbl['trans_'+tool] = np.sqrt(new_tbl[tool + '-norm'])
        elif transform == 'pwr':
            new_tbl['trans_'+tool] = new_tbl[tool + '-norm']**2
        elif transform == 'multi':
            new_tbl['trans_'+tool] = (new_tbl[tool + '-norm'])*2
        elif transform == 'arcsin':
            new_tbl['trans_'+tool] = np.arcsin(new_tbl[tool + '-norm'])
        new_tbl['trans_'+tool+'_norm'] = minmaxscale(new_tbl['trans_' + tool])
        tool = 'trans_'+tool

    inner_min, inner_max = get_new_minmax(new_tbl[tool])
    outer_min, outer_max = get_new_minmax(new_tbl[tool], "outer")

    new_tbl[tool + '-adj-1'] = [clip_outliers(i, inner_min, inner_max) for i in new_tbl[tool]]
    new_tbl[tool + '-adj-1-norm'] = minmaxscale(new_tbl[tool + '-adj-1'])
    new_tbl[tool + '-adj-2'] = [clip_outliers(i, outer_min, outer_max) for i in new_tbl[tool]]
    new_tbl[tool + '-adj-2-norm'] = minmaxscale(new_tbl[tool + '-adj-2'])
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

def create_normalised_sent_table(src_tbl, tool, transform=None):
    new_tbl = src_tbl[['id', 'comments', tool]]
    new_tbl.is_copy = False

    new_tbl[tool + '-norm'] = minmaxscale_sent(new_tbl,tool, True)

    if transform != None:
        new_tbl['trans_'+tool] = transform_sent(new_tbl, tool + '-norm', transform)
        new_tbl['trans_'+tool+'_norm'] = minmaxscale_sent(new_tbl, 'trans_' + tool)
        tool = 'trans_'+tool

    inner_min, inner_max = get_new_minmax_sent(new_tbl[tool])
    outer_min, outer_max = get_new_minmax_sent(new_tbl[tool], "outer")
    new_tbl[tool + '-adj-1'] = [clip_outliers_sent(i, inner_min, inner_max) for i in new_tbl[tool]]
    new_tbl[tool + '-adj-1-norm'] = minmaxscale_sent(new_tbl, tool + '-adj-1')
    new_tbl[tool + '-adj-2'] = [clip_outliers_sent(i, outer_min, outer_max) for i in new_tbl[tool]]
    new_tbl[tool + '-adj-2-norm'] = minmaxscale_sent(new_tbl, tool + '-adj-2')
    return new_tbl

def dic_to_list(dic, l):
    if l != len(dic):
        return np.nan
    return [dic[str(i)] for i in range(l)]
def str_to_list(x):
    return np.array(x.replace('\'', '').replace('[', '').replace(']', '').split(", ")).astype(np.float)

def flatten(lists, convert=True):
    return [float(v) for l in lists for v in l] if convert else [v for l in lists for v in l]

def hasNumbers(x):
    return bool(re.search(r'\d', x))

def invert_dict(d):
    return dict([ (v, k) for k, v in d.items( ) ])

def convert_to_str(ls):
    return [str(i) if i != None else None for i in ls]

 
