from IPython.core.display import HTML, display_html
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
import utils
from collections import defaultdict
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import seaborn as sns

def rss(tbl, c1, c2):
    return np.sum([(float(r[c1]) - float(r[c2]))**2 for i, r in tbl.iterrows()])

def split_into_levels(in_tbl, out_tbl, col, type=1):
    vals = []
    out_tbl.is_copy = False
    if type == 1:
        for i, r in in_tbl.iterrows():
            val = float(r[col])
            if (val<=0.2):
                vals.append(1)
            elif (val>0.2 and val<=0.4):
                vals.append(2)
            elif (val>0.4 and val<=0.6):
                vals.append(3)
            elif (val>0.6 and val<=0.8):
                vals.append(4)
            elif (val>0.8):
                vals.append(5)
    elif type == 2:
        for i, r in in_tbl.iterrows():
            val = float(r[col])
            if (val<=0.35):
                vals.append(1)
            elif (val>0.35 and val<=0.65):
                vals.append(2)
            elif (val>0.65):
                vals.append(3)
    else:
        return none
    if type == 1:
        out_tbl["level5_" + col] = vals
    else:
        out_tbl["level3_" + col] = vals
    return out_tbl

def build_levels_table(tbl, tools, type=1):
    levels_tbl = tbl[['id', 'comments']]
    for t in tools:
        levels_tbl = split_into_levels(tbl, levels_tbl, t, type)
    return levels_tbl

def create_confusion_matrix(labels, predictions):
    """
    Creates a confusion matrix that counts for each gold label how often it was labelled by what label
    in the predictions.
    Args:
        data: a list of gold (x,y) pairs.
        predictions: a list of y labels, same length and with matching order.

    Returns:
        a `defaultdict` that maps `(gold_label,guess_label)` pairs to their prediction counts.
    """
    confusion = defaultdict(int)
    for y_gold, y_guess in zip(labels, predictions):
        confusion[(y_gold, y_guess)] += 1
    return confusion

def analyse_review (text, scores, type=1):
    sents = sent_tokenize(text)
    out = []
    if type == 1:
        for i, sent in zip(range(len(sents)), sents):
            if (scores[i]<=0.2):
                out.append("<font color='#c40909'>" + sent +"</font>")
            elif (scores[i]>0.2 and scores[i]<=0.4):
                out.append("<font color='#ef8c13'>" + sent +"</font>")
            elif (scores[i]>0.4 and scores[i]<=0.6):
                out.append("<font color='#ecef13'>" + sent +"</font>")
            elif (scores[i]>0.6 and scores[i]<=0.8):
                out.append("<font color='#85c433'>" + sent +"</font>")
            elif (scores[i]>0.8):
                out.append("<font color='#4a9b37'>" + sent +"</font>")
    elif type == 2:
        for i, sent in zip(range(len(sents)), sents):
            if (scores[i]<=0.35):
                out.append("<font color='#c40909'>" + sent +"</font>")
            elif (scores[i]>0.35 and scores[i]<=0.65):
                out.append("<font color='#ecef13'>" + sent +"</font>")
            elif (scores[i]>0.65):
                out.append("<font color='#4a9b37'>" + sent +"</font>")
    else:
        return None

    words = "<div style='margin: 20px;border: solid 20px;'><p style='background-color:black;margin=10px;'><b>" + " ".join(out) + "</b></p></div>"
    return HTML(words)

def plot_confusion_matrix_dict(matrix_dict, rotation=45, outside_label="", t=""):
    labels = set([y for y, _ in matrix_dict.keys()] + [y for _, y in matrix_dict.keys()])
    sorted_labels = sorted(labels)
    matrix = np.zeros((len(sorted_labels), len(sorted_labels)))
    for i1, y1 in enumerate(sorted_labels):
        for i2, y2 in enumerate(sorted_labels):
            if y1 != outside_label or y2 != outside_label:
                matrix[i1, i2] = matrix_dict[y1, y2]
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(sorted_labels))
    plt.xticks(tick_marks, sorted_labels, rotation=rotation)
    plt.yticks(tick_marks, sorted_labels)
    plt.xlabel(t)
    # plt.tight_layout()
    # plt.suptitle(t)

def evaluate(conf_matrix, label_filter=None):
    """
    Evaluate Precision, Recall and F1 based on a confusion matrix as produced by `create_confusion_matrix`.
    Args:
        conf_matrix: a confusion matrix in form of a dictionary from `(gold_label,guess_label)` pairs to counts.
        label_filter: a set of gold labels to consider. If set to `None` all labels are considered.

    Returns:
        Precision, Recall, F1 triple.
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for (gold, guess), count in conf_matrix.items():
        if label_filter is None or gold in label_filter or guess in label_filter:
            if gold == 'None' and guess != gold:
                fp += count
            elif gold == 'None' and guess == gold:
                tn += count
            elif gold != 'None' and guess == gold:
                tp += count
            elif gold != 'None' and guess == 'None':
                fn += count
            else:  # both gold and guess are not-None, but different
                fp += count if label_filter is None or guess in label_filter else 0
                fn += count if label_filter is None or gold in label_filter else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec * recall > 0 else 0.0
    return prec, recall, f1

def plot_confusion_matrix_grid(tbl, true_label_col, guess_labels):
    c = 1
    plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    for i, t in zip(range(1, len(guess_labels) +1), guess_labels):
        cm = create_confusion_matrix(tbl[true_label_col], tbl[t])
        plt.subplot(4 , 4, c)
        plot_confusion_matrix_dict(cm, t=t)
        if c % 4 == 0:
            c = 1
            plt.show()
            plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
        elif i != len(guess_labels):
            c = c  + 1
        else:
            plt.show()
        print(t)
        display_html(full_evaluation_table(cm))

def full_evaluation_table(confusion_matrix):
    """
    Produce a pandas data-frame with Precision, F1 and Recall for all labels.
    Args:
        confusion_matrix: the confusion matrix to calculate metrics from.

    Returns:
        a pandas Dataframe with one row per gold label, and one more row for the aggregate of all labels.
    """
    labels = sorted(list({l for l, _ in confusion_matrix.keys()} | {l for _, l in confusion_matrix.keys()}))
    gold_counts = defaultdict(int)
    guess_counts = defaultdict(int)
    for (gold_label, guess_label), count in confusion_matrix.items():
        if gold_label != "None":
            gold_counts[gold_label] += count
            gold_counts["[All]"] += count
        if guess_label != "None":
            guess_counts[guess_label] += count
            guess_counts["[All]"] += count

    result_table = []
    for label in labels:
        if label != "None":
            result_table.append((label, gold_counts[label], guess_counts[label], *evaluate(confusion_matrix, {label})))

    result_table.append(("[All]", gold_counts["[All]"], guess_counts["[All]"], *evaluate(confusion_matrix)))
    return pd.DataFrame(result_table, columns=('Label', 'Gold', 'Guess', 'Precision', 'Recall', 'F1'))

def get_dummies(data, nominal_columns):
    dummy_df = pd.get_dummies(data[nominal_columns])
    data = pd.concat([data, dummy_df], axis=1)
    data = data.drop(nominal_columns, axis=1)
    return data

def print_coeffcorr(X):

    ## Correlation confusion matrix
    corr = X.corr()
    sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
    plt.show()

    # Correlation matrix with values
    display_html(X.corr().style.background_gradient())


def analyse_features(X, Y, thresh=10):

    print_coeffcorr(X)

    # Run VIF
    X = calculate_vif_(X, thresh)

    run_regression(X, Y)

def calculate_vif_(X, thresh=10):
    cols = X.columns
    variables = np.arange(X.shape[1])
    dropped=True
    while dropped:
        dropped=False
        c = X[cols[variables]].values
        vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]
        print (pd.DataFrame({"vif": vif, "cols":cols[variables] }))
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X[cols[variables]].columns[maxloc] + '\' at index: ' + str(maxloc))
            variables = np.delete(variables, maxloc)
            dropped=True

    print('Remaining variables:')
    print(X.columns[variables])
    return X[cols[variables]]

def run_regression(X, y):
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    print(model.summary())

def test_features(X, Y, nominal_features=None, vif_thresh=10):

    #data_Y_sat = data_X.join(Y[['id','satisfaction']] .set_index('id'), on='id').dropna()
    #data_Y_rel_sat = data_X.join(Y[['id','relative_satisfaction']] .set_index('id'), on='id').dropna()

    feature_labels = set(X.columns)

    analyse_features(X, Y, vif_thresh)
    #analyse_features(X, [data_Y_sat['satisfaction'], data_Y_rel_sat['relative_satisfaction']], vif_thresh)
