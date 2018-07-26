import sys
sys.path.append("../../Code")
import utils
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

stopWords = set(stopwords.words('english'))

with open('../../Tools/opinion-lexicon-English/positive-words.txt', 'r') as f:
    positive_words = f.read().splitlines()

with open('../../Tools/opinion-lexicon-English/negative-words.txt', 'r') as f:
    negative_words = f.read().splitlines()

positive_words_set = set(positive_words)
negative_words_set = set(negative_words)

def get_score(text, type="bow"):
    tokenized = utils.tokenize(text.lower())
    pos_score = 0
    neg_score = 0

    if type == "bow":
        word_set = set(tokenized) - stopWords
        pos_score = len(word_set.intersection(positive_words_set))/len(word_set) if len(word_set) > 0 else 0
        neg_score = len(word_set.intersection(negative_words_set))/len(word_set) if len(word_set) > 0 else 0
    elif type == "freq":
        for word in tokenized:
            if word not in stopWords:
                if word in positive_words_set:
                    pos_score += 1
                if word in negative_words_set:
                    neg_score += 1
        pos_score = pos_score/len(tokenized) if len(tokenized) > 0 else 0
        neg_score = neg_score/len(tokenized) if len(tokenized) > 0 else 0
    else:
        return np.NaN
    return pos_score - neg_score

def get_sentence_level_sent(text, type='bow'):
    sents = sent_tokenize(text)
    scores = []
    for s in sents:
        scores.append(get_score(s, type))
    return scores
