from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

def get_sentence_level_sent(text):
    analyzer = SentimentIntensityAnalyzer()
    sents = sent_tokenize(text)
    scores = []
    for s in sents:
        scores.append(analyzer.polarity_scores(s)['compound'])
    return scores

def get_score(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)['compound']
