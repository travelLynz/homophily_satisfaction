from googletrans import Translator
from langdetect import detect, detect_langs
import nltk
import re
import emoji
import time
import numpy as np
import os
import pandas as pd
import utils
import profiles
from nltk.tokenize import sent_tokenize, WhitespaceTokenizer, WordPunctTokenizer
import nltk.classify.textcat as tc
import collections as ct

with open('../../Tools/AFINN-165.txt', 'r') as f:
    afinn_scores = {}
    for line in f:
        text = line.replace('\n', ' ').split("\t")
        afinn_scores[text[0]] = int(text[1])
afinn_score_dict = dict(nltk.Index((value, key) for (key,value) in afinn_scores.items()))
pos_words = set(afinn_score_dict[2]+afinn_score_dict[3]+afinn_score_dict[4]+afinn_score_dict[5])
neg_words = set(afinn_score_dict[-2]+afinn_score_dict[-3]+afinn_score_dict[-4]+afinn_score_dict[-5])

def get_langdetect_languages(tbl, col):
    tbl.is_copy = False
    langs = []
    lang_conf = []
    for s in tbl[col]:
        try:
            first = str(detect_langs(s)[0]).split(':')
            langs.append(first[0])
            lang_conf.append(float(first[1]))
        except :
            langs.append('unk')
            lang_conf.append(0.0)
    tbl['langdetect_langs'], tbl['langdetect_langs_conf'] = langs, lang_conf
    return tbl

def get_token_len(text):
    token_text = utils.tokenize(text) if text not in [np.nan, None ] and len(text) > 0 else []
    return len(token_text)

def get_sent_len(text):
    sent_text = sent_tokenize(text) if text not in [np.nan, None ] and len(text) > 0 else []
    return len(sent_text)

def get_textcat_languages(tbl, col):
    t = tc.TextCat()
    tbl.is_copy = False
    langs = []
    for s in tbl[col]:
        try:
            l = t.guess_language(s)
            langs.append(l)
        except :
            langs.append('unk')
    tbl['textcat_langs'] = langs
    return tbl

def get_google_languages(tbl, col):
    tbl.is_copy = False
    langs = []
    lang_conf = []
    for  i, r in tbl.iterrows():
        # if (i+1) % 250 == 0:
        #     time.sleep(60)
        t = Translator()
        try:
            result = t.detect(str(r[col]))
            langs.append(result.lang)
            lang_conf.append(result.confidence)
        except ValueError :
            langs.append('unk')
            lang_conf.append(0.0)
        except Error as  e:
            langs.append('err')
            lang_conf.append(0.0)
            print(e)
    tbl['google_langs'], tbl['google_langs_conf'] = langs, lang_conf
    return tbl

def get_other_langs(table):
    table.is_copy = False
    translated = [detect_other_langs(r['comments']) for i, r in table.iterrows()]
    table['other_langs'] = [x[0] for x in translated]
    table['translated'] = [x[1] for x in translated]
    return table

def remove_emoticons(text):
    return emojis.emoji_pattern.sub(r'', text)

def give_emoji_free_text(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    return clean_text

def split_get_google_languages_files(tbl, col, size, dir='.', start=0):
    for n, data in zip(range(start,size),np.array_split(tbl, size)[start:]):
        din = os.path.join(dir, 'google_trans'+ "_" + str(n) + '.csv')
        result = get_google_languages(data, col)
        result.to_csv(din)

def split_get_google_languages(tbl, col, size, start=0):
    results = []
    for n, data in zip(range(start,size),np.array_split(tbl, size)[start:]):
        results.append(get_google_languages(data, col))
    return pd.concat(results)

def detect_other_langs(com):
    sents = sent_tokenize(com)
    other_langs = []
    for i,s in zip(range(len(sents)), sents):
        try:
            first = str(detect_langs(s)[0]).split(':')
            if len(first)> 1 and first[0] != 'en' and float(first[1]) > 0.999:
                other_langs.append(first[0])
                translator = Translator()
                translated = translator.translate(s, dest='en')
                sents[i] = translated.text
        except:
            continue;
    return (other_langs, "".join(sents)) if len(other_langs) > 0 else (0, None)

def translate_text(text):
    translator = Translator()
    translated = translator.translate(str(text), dest='en')
    return translated.text

def get_english_text(tbl, col):
    updated_text = []
    for i, r in tbl.iterrows():
        if (r['google_langs'] == 'unk'):
            updated_text.append(r[col])
        elif (r['google_langs'] != 'en') or (float(str(r['google_langs_conf'])) < 1.0):
            print("Here")
            updated_text.append(translate_text(str(r[col])))
        else:
            updated_text.append(r[col])
    return updated_text

def pipeline(data, col, stopwords=[], vocab=None, encode_data=True):
    if encode_data:
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
        if encode_data:
            data_encoded[d['id']] = r
    N = len(data)
    idf_dict = []
    for key in vocab.keys():
        if key in vocab_doc_count.keys():
            idf_dict.append(np.log10(N/vocab_doc_count[key]))
        else:
            idf_dict.append(0)
    return (data_encoded, vocab_counts, vocab, idf_dict) if encode_data else (vocab_counts, vocab, idf_dict)

def create_vocab_count_table(counts):
    return pd.DataFrame({'counts':[counts[k] for k in counts.keys()], 'word':[k for k in counts.keys()]})

def extract_known_words(words, w2v, separate_numbers=False):
    known_words = []
    unknown_words = []
    numbered = []

    for word in words:
        try:
            _ = w2v[word]
            known_words.append(word)
        except:
            if not separate_numbers or not utils.hasNumbers(word):
                unknown_words.append(word)
            else:
                numbered.append(word)
    return (known_words, unknown_words) if not separate_numbers else (known_words, unknown_words, numbered)

def get_personality_metrics(text):
    if(text in [np.nan, None] or len(text) == 0):
        return None
    avg_wc = np.mean([len(utils.tokenize(t)) for t in text])
    text = str(" ".join(text))
    token_text = utils.tokenize(text)
    if(len(token_text) == 0):
        return None
    sent_token_text = sent_tokenize(text)
    pos_tags =  nltk.pos_tag(token_text)
    pos_dict = dict(nltk.Index((value, key) for (key,value) in pos_tags))

    empath_dict = profiles.get_empath(text)
    result = {}

    punct = ".?!("
    cd = {c:val for c, val in ct.Counter(text).items() if c in punct}

    negations = set(['no', 'not', 'none','nobody', 'nothing', 'neither', 'nowhere', 'never'])
    tentative = set(['can', 'may','might', 'could', 'should', 'will', 'would', 'possibly', 'probably', 'likely', 'suggest', 'appear', 'indicate'])
    articles = ['DT']
    adverbs = ['RB', 'RBR','RBS']
    adjectives = ['JJ', 'JJR', 'JJS']
    verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP','VBZ']
    nouns = ['NN', 'NNS']
    pronouns = ['NNP', 'NNPS']
    interjections = ['UH']
    prepositions = ['IN']
    when_words = ['WDT', 'WP', 'WP$', 'WRB']
    conjunctions = ['CC']
    modal_words = ['MD']
    foreign_words = ['FW']
    other_pronouns = ['PRP$']
    particles = ['RP']
    numbers  = ['CD']
    to  = ['TO']


    articles_freq = get_freq(articles, pos_dict)
    adverbs_freq = get_freq(adverbs, pos_dict)
    adjectives_freq = get_freq(adjectives, pos_dict)
    verbs_freq = get_freq(verbs, pos_dict)
    nouns_freq = get_freq(nouns, pos_dict)
    pronouns_freq = get_freq(pronouns, pos_dict)
    interjections_freq = get_freq(interjections, pos_dict)
    prepositions_freq = get_freq(prepositions, pos_dict)

    tokens_lower = [i.lower() for i in token_text]
    unique_words = set(tokens_lower)
#     emo_tokens = [np.sign(af.get_score(t)) for t in unique_words]
    pro_lower = [i.lower() for i in pos_dict['PRP']] if 'PRP' in pos_dict.keys() else []


    word_lengths = [len(w) for w in token_text]
    result['avg_wc'] = avg_wc
    result['avg_word_len'] = np.mean(word_lengths)
    result['avg_sent_len'] = np.mean([len(s) for s in sent_token_text])
    result['avg_long_words'] = np.sum([1 for w in word_lengths if w > 6])/len(token_text)
    #result['avg_nums'] =  get_freq(numbers, pos_dict)/len(token_text)
    result['articles'] = articles_freq/len(token_text)
    result['adverbs'] = adverbs_freq/len(token_text)
    result['adjectives'] = adjectives_freq/len(token_text)
    result['verbs'] = verbs_freq/len(token_text)
    result['nouns'] = nouns_freq/len(token_text)
    result['pronouns'] = pronouns_freq/len(token_text)
    result['interjections'] = interjections_freq/len(token_text)
    #result['prepositions'] = prepositions_freq/len(token_text)
    result['i'] = pro_lower.count("i")/len(token_text)
    result['we'] = pro_lower.count("we")/len(token_text)
    result['you'] = pro_lower.count("you")/len(token_text)
    result['self'] = np.sum([1 for i in pro_lower if i.endswith("self")])/len(token_text)
    result['pos_count'] = len(pos_dict.keys())
    result['unique_words'] = len(unique_words)/len(token_text)
    result['negations'] = np.sum([1 for t in token_text if t.lower() in negations])/len(token_text)
    result['tentative'] = np.sum([1 for t in token_text if t.lower() in tentative])/len(token_text)
    result['formality'] = (nouns_freq + adjectives_freq + prepositions_freq + articles_freq - pronouns_freq - verbs_freq - adverbs_freq - interjections_freq + 100)/2
    #result['when'] = get_freq(when_words, pos_dict)/len(token_text)
    result['conj'] = get_freq(conjunctions, pos_dict)/len(token_text)
    #result['foreign'] = get_freq(foreign_words, pos_dict)/len(token_text)
    result['other_pro'] = get_freq(other_pronouns, pos_dict)/len(token_text)
    result['to'] = get_freq(to, pos_dict)/len(token_text)
    result['modal'] = get_freq(modal_words, pos_dict)/len(token_text)
    result['swear'] = empath_dict['swearing_terms']
    #result['particles'] = get_freq(particles, pos_dict)/len(token_text)
    result['capital'] = np.sum([1 for w in token_text if w.isupper()])/len(token_text)
    result['positive_words'] = np.sum([tokens_lower.count(w) for w in pos_words])/len(tokens_lower)
    result['negative_words'] = np.sum([tokens_lower.count(w) for w in neg_words])/len(tokens_lower)

    return result

def get_freq(group, dic):
    res = 0
    for g in group:
        if g in dic.keys():
            res += len(dic[g])
    return res

def get_personality_metrics_reviews(review, num_revs):
    results = get_personality_metrics(review)
    if results == None or len(results.keys()) == 0 :
        return None
    else:
        for k in results.keys():
            results[k] /= num_revs
    return results

def correct_text(text, cdict):
    w = WhitespaceTokenizer()
    p = WordPunctTokenizer()
    token = w.tokenize(text=text)
    for i,s in enumerate(token):
        split = p.tokenize(s)
        for j,e in enumerate(split):
            if e in set(cdict.keys()):
                split[j] = cdict[e]
        token[i] = "".join(split)
    return " ".join(token)

def spell_correct(cdict, src_tbl, col):
    corrected = []
    corrected_text = []
    for i, r in src_tbl.iterrows():
        corrections = set(utils.tokenize(str(r[col]))).intersection(set(cdict.keys()))
        needs_correction = 1 if len(corrections) > 0 else 0
        corrected_text.append(correct_text(r[col], cdict) if needs_correction else r[col])
        corrected.append(needs_correction)
    src_tbl.is_copy = False
    src_tbl['isCorrected'] = corrected
    src_tbl[col] = corrected_text
    return src_tbl
