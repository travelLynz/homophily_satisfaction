from IPython.core.display import HTML, display_html
from nltk.tokenize import sent_tokenize
import utils
import numpy as np
def analyse_review (text, scores, type=1):
    sents = sent_tokenize(text)
    out = []
    if type == 1:
        for i, sent in zip(range(len(sents)), sents):
            if (float(scores[i])<=0.2):
                out.append("<font color='#c40909'>" + sent +"</font>")
            elif (float(scores[i])>0.2 and float(scores[i])<=0.4):
                out.append("<font color='#ef8c13'>" + sent +"</font>")
            elif (float(scores[i])>0.4 and float(scores[i])<=0.6):
                out.append("<font color='#ecef13'>" + sent +"</font>")
            elif (float(scores[i])>0.6 and float(scores[i])<=0.8):
                out.append("<font color='#85c433'>" + sent +"</font>")
            elif (float(scores[i])>0.8):
                out.append("<font color='#4a9b37'>" + sent +"</font>")
    elif type == 2:
        for i, sent in zip(range(len(sents)), sents):
            if (float(scores[i])<=0.35):
                out.append("<font color='#c40909'>" + sent +"</font>")
            elif (float(scores[i])>0.35 and float(scores[i])<=0.65):
                out.append("<font color='#ecef13'>" + sent +"</font>")
            elif (float(scores[i])>0.65):
                out.append("<font color='#4a9b37'>" + sent +"</font>")
    else:
        return None

    words = "<div style='margin: 20px;border: solid 20px;'><p style='background-color:black;margin=10px;'><b>" + " ".join(out) + "</b></p></div>"
    return HTML(words)

def reduce_to_average(col):
    vals = []
    for i in col:
        reduced = [v for v in utils.to_float(i) if v <= 0.45 or v >= 0.55]
        vals.append(format(np.average(reduced), ".3f") if len(reduced) > 0 else 0.5)
    return vals

def reduce_sentence_scores(table, tools, isString=False):
    table.is_copy = False
    for t in tools:
        if isString:
            table[t] = table[t].map(lambda x : np.array(x.replace('\'', '').replace('[', '').replace(']', '').split(", ")).astype(np.float))
        table[t] = reduce_to_average(table[t])
    return table

def print_anaysis(tbl, id, scoring, type=1):
    display_html(analyse_review(utils.get_comments(tbl, 'id', id , 'comments'), list(tbl[tbl['id'] == id ][scoring].values[0]), type))
