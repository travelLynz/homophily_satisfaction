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

def minmaxscale_sent(tbl, col):
    val_set = set([])
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

def to_float(a):
    return [float(i) for i in a ]

def clip_outliers(x, low, high):
    if x < low:
        return low
    elif x> high:
        return high
    else:
        return x

def get_new_minmax(vals, fence_type="inner"):
    q1 = np.percentile(vals, 25)
    q3 = np.percentile(vals, 75)
    iqr = q3 - q1
    margin = iqr*1.5 if fence_type == 'inner' else iqr*3
    min = q1 - margin
    max = q3 + margin
    return (min, max)

def create_normalised_table(src_tbl, tool, transform=False):
    new_tbl = src_tbl[['id', 'comments', tool]]
    new_tbl.is_copy = False

    new_tbl[tool + '-norm'] = minmaxscale(new_tbl[tool])

    if transform:
            new_tbl['trans_'+tool] = 2**new_tbl[tool]
            new_tbl['trans_'+tool+'_norm'] = minmaxscale(new_tbl['trans_' + tool])
            tool = 'trans_'+tool

    inner_min, inner_max = get_new_minmax(new_tbl[tool])
    outer_min, outer_max = get_new_minmax(new_tbl[tool], "outer")

    new_tbl[tool + '-adj-1'] = [clip_outliers(i, inner_min, inner_max) for i in new_tbl[tool]]
    new_tbl[tool + '-adj-1-norm'] = minmaxscale(new_tbl[tool + '-adj-1'])
    new_tbl[tool + '-adj-2'] = [clip_outliers(i, outer_min, outer_max) for i in new_tbl[tool]]
    new_tbl[tool + '-adj-2-norm'] = minmaxscale(new_tbl[tool + '-adj-2'])
    return new_tbl

def dic_to_list(dic):
    return [dic[str(i)] for i in range(len(dic))]
