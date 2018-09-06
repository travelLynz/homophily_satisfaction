from empath import Empath
import numpy as np
import pandas as pd

def create_empath_dict(tbl, col):
    return {r['id']:np.array(list(get_empath(r[col], normalize = True).values())) for _, r in tbl.iterrows()}

def get_empath(text, normalize = False):
    lexicon = Empath()
    res = lexicon.analyze(str(text).lower(), normalize=normalize)
    return res if res is not None else {}

def cosine_similarity(vec_x, vec_y):
    return (vec_x.T.dot(vec_y))/(np.sqrt(np.sum(vec_x)**2)*np.sqrt(np.sum(vec_y)**2))

def add_cosine_similarity(table, vec, version="", id_type="str"):
    cosine = []
    if 'cosine_similarity'+version in table.columns:
        table.drop(['cosine_similarity'+version], axis=1)
    cosine = [cosine_similarity(vec[str(r['recipient_id'])], vec[str(r['reviewer_id'])]) if id_type == "str" else cosine_similarity(vec[r['recipient_id']], vec[r['reviewer_id']])  for i, r in table.iterrows() ]
    table = table.join(pd.DataFrame({'cosine_similarity'+version:cosine}).reset_index(drop=True))
    return table

def bag_of_words(sent, vocab_length, vectype='one-hot'):
    words = []
    rep = np.zeros(vocab_length)
    for w in sent:
        if vectype == 'one-hot':
            if w not in words:
                rep[w] += 1
                words.append(w)
        else:
            rep[w] += 1
    return rep

def create_representation_bow(data, vocab_length, vectype='one-hot'):
    rep = dict()
    for key in data.keys():
        rep[key] = bag_of_words(data[key], vocab_length, vectype)
    return rep

def add_kl_divergence(table, hlms, alms, version=""):
    if 'kl_divergence'+ version in table.columns:
        table.drop(['kl_divergence'+version], axis=1)
    kl = [calculate_KL_divergence(hlms[h], alms[a]) for h, a in zip(table.Headline, table['Body ID'])]
    table = table.join(pd.DataFrame({'kl_divergence+'+version:kl}).reset_index(drop=True))
    return table

def add_tfidf(table, pvec, pfreq, idf, version=""):
    if 'tfidf'+version in table.columns:
        table.drop(['tfidf'+version], axis=1)
    tfidf = [calculate_tfidf(str(g), str(h), pvec, pfreq, idf) for g, h in zip(table['reviewer_id'], table['recipient_id'])]
    table = table.join(pd.DataFrame({'tfidf'+version:tfidf}).reset_index(drop=True))
    return table

def create_lm_dict(data, col, vocab):
    epsilon = 0.0000001
    lms = {}
    for _, k in data.iterrows():
        lm = build_lm_dict(k[col])
        lms[k['id']] = [lm.probability(w) + epsilon for w in vocab]
    return lms

import abc
import math
import collections
class CountLM(metaclass=abc.ABCMeta):
    """
    A Language Model that uses counts of events and histories to calculate probabilities of words in context.
    """
    def __init__(self, vocab, order):
        self.vocab = vocab
        self.order = order

    @abc.abstractmethod
    def counts(self, word_and_history):
        pass

    @abc.abstractmethod
    def norm(self, history):
        pass

    def probability(self, word, *history):
        if word not in self.vocab:
            return 0.0
        sub_history = tuple(history[-(self.order - 1):]) if self.order > 1 else ()
        norm = self.norm(sub_history)
        if norm == 0:
            return 1.0 / len(self.vocab)
        else:
            return self.counts((word,) + sub_history) / self.norm(sub_history)

class NGramLM(CountLM):
    def __init__(self, train, order):
        """
        Create an NGram language model.
        Args:
            train: list of training tokens.
            order: order of the LM.
        """
        super().__init__(set(train), order)
        self._counts = collections.defaultdict(float)
        self._norm = collections.defaultdict(float)
        for i in range(self.order, len(train)):
            history = tuple(train[i - self.order + 1: i])
            word = train[i]
            self._counts[(word,) + history] += 1.0
            self._norm[history] += 1.0

    def counts(self, word_and_history):
        return self._counts[word_and_history]

    def norm(self, history):
        return self._norm[history]

def calculate_KL_divergence(pmh, pma):
    return -np.sum(pmh*np.log(pma))

def build_lm_dict(text, ngram=2, alpha=0.1):
    return LaplaceLM(NGramLM(utils.tokenize(text), ngram), alpha)

class LaplaceLM(CountLM):
    def __init__(self, base_lm, alpha):
        super().__init__(base_lm.vocab, base_lm.order)
        self.base_lm = base_lm
        self.alpha = alpha

    def counts(self, word_and_history):
        return self.base_lm.counts(word_and_history) + self.alpha

    def norm(self, history):
        return self.base_lm.norm(history) + self.alpha * len(self.base_lm.vocab)

OOV = "<OOV>"
def inject_OOVs(data):
    """
    Uses a heuristic to inject OOV symbols into a dataset.
    Args:
        data: the sequence of words to inject OOVs into.

    Returns: the new sequence with OOV symbols injected.
    """
    seen = set()
    result = []
    for word in data:
        if word in seen:
            result.append(word)
        else:
            result.append(OOV)
            seen.add(word)
    return result

def calculate_tfidf(g_key, h_key, pvec, pfreq, idf):
    return np.sum(pvec[g_key]*pvec[h_key]*pfreq[h_key]*idf)

def pipeline(data, col, stopwords=[], vocab=None):
    data_encoded = {}
    vocab_counts = {}
    vocab_doc_count = {}

    is_ext_vocab = True
    if vocab is None:
        is_ext_vocab = False
        vocab = {'<OOV>': 0}
    for _,d in data.iterrows():
        r = []
        doc_vocab = set()
        tokenized_r = utils.tokenize(d[col])
        for token in tokenized_r:
            #token = ps.stem(token)
            if token.lower() in stopwords:
                continue
            if not is_ext_vocab and token not in vocab:
                vocab[token] = len(vocab)
                vocab_counts[token] = 1
                doc_vocab.add(token)
                vocab_doc_count[token] = 1
            if token not in vocab:
                token_id = vocab['<OOV>']
                vocab_counts['<OOV>'] += 1
            elif token not in doc_vocab:
                doc_vocab.add(token)
                vocab_doc_count[token] = 1 if token not in vocab_doc_count.keys() else vocab_doc_count[token] + 1
                token_id = vocab[token]
                vocab_counts[token] += 1
            else:
                token_id = vocab[token]
                vocab_counts[token] += 1
            r.append(token_id)
        data_encoded[d['id']] = r
    N = len(data)
    idf_dict = []
    for key in vocab.keys():
        if key in vocab_doc_count.keys():
            idf_dict.append(np.log10(N/vocab_doc_count[key]))
        else:
            idf_dict.append(0)
    return (data_encoded, vocab_counts, vocab, idf_dict)

def summarize_host_table(trip_table, country_info):
    host_ids = trip_table['hostId'].unique()
    countries_visited = []
    regions_visited = []
    sub_regions_visited = []
    for h in host_ids:
        c_v, r_v, sr_v = {}, {}, {}
        hset = trip_table[trip_table['hostId'] == h]
        for _, r in hset.iterrows():
            if r['ccode'] != 'UNK':
                cnty = r['ccode']
                reg = list(country_info[country_info['alpha2Code'] == cnty]['region'])[0]
                sreg = list(country_info[country_info['alpha2Code'] == cnty]['subregion'])[0]
                c_v[cnty] = c_v[cnty] + r['min_times'] if cnty in c_v.keys() else r['min_times']
                r_v[reg] = r_v[reg] + r['min_times'] if reg in r_v.keys() else r['min_times']
                sr_v[sreg] = sr_v[sreg] + r['min_times'] if sreg in sr_v.keys() else r['min_times']
            else:
                c_v['UNK'] = r['min_times'] if "UNK" in c_v.keys() else r['min_times']
                r_v['UNK'] = r['min_times'] if "UNK" in r_v.keys() else r['min_times']
                sr_v['UNK'] = r['min_times'] if "UNK" in sr_v.keys() else r['min_times']
        countries_visited.append(c_v)
        regions_visited.append(r_v)
        sub_regions_visited.append(sr_v)
    new_tbl = pd.DataFrame({'id': list(host_ids), 'countries_visited': countries_visited, 'regions_visited': regions_visited , 'sub_regions_visited': sub_regions_visited })
    return new_tbl

def print_country_table(reviews, guests, col):
    unq = list(guests[col].unique())
    g_cnts = []
    r_cnts = []
    stds = []
    means = []
    for v in unq:
        g = guests[guests[col] == v] if v not in [np.nan] else guests[guests[col].isnull()]
        revs = reviews[reviews['reviewer_id'].isin(g['id'].unique())]
        g_cnts.append(len(revs['reviewer_id'].unique()))
        r_cnts.append(len(revs))
        means.append(np.mean(revs['satisfaction']))
        stds.append(np.std(revs['satisfaction']))
    return pd.DataFrame({col:unq, 'guest_count':g_cnts, 'review_count': r_cnts, 'average':means, 'stddev':stds})

def print_country_table_simple(reviews, guests, col, listings):
    names = ["Same", "Different"]
    g_cnts = []
    r_cnts = []
    means= []
    stds = []
    v = 'Americas' if col == 'region' else 'Northern America'

    same_g = guests[guests[col] == v]
    revs = reviews[(reviews['reviewer_id'].isin(same_g['id'].unique())) & (reviews['listing_id'].isin(listings['id'].unique()))]
    g_cnts.append(len(revs['reviewer_id'].unique()))
    r_cnts.append(len(revs))
    means.append(np.mean(revs['satisfaction']))
    stds.append(np.std(revs['satisfaction']))

    diff_g = guests[guests[col] != v]
    revs = reviews[(reviews['reviewer_id'].isin(diff_g['id'].unique())) & (reviews['listing_id'].isin(listings['id'].unique()))]
    g_cnts.append(len(revs['reviewer_id'].unique()))
    r_cnts.append(len(revs))
    means.append(np.mean(revs['satisfaction']))
    stds.append(np.std(revs['satisfaction']))

    return pd.DataFrame({col:names, 'guest_count':g_cnts, 'review_count': r_cnts, 'average':means, 'stddev':stds})
