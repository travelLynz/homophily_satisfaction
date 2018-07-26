from afinn import Afinn
from nltk.tokenize import sent_tokenize

def get_sentence_level_sent(text):
    afinn = Afinn(emoticons=True)
    sents = sent_tokenize(text)
    scores = []
    for s in sents:
        scores.append(afinn.score(s))
    return scores

def get_score(text):
    afinn = Afinn(emoticons=True)
    return afinn.score(text)
