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
from nltk.tokenize import sent_tokenize
import nltk.classify.textcat as tc

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
    t = Translator()
    tbl.is_copy = False
    langs = []
    lang_conf = []
    for  i, r in tbl.iterrows():
        # if (i+1) % 250 == 0:
        #     time.sleep(60)
        try:
            result = t.detect(r[col])
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
    translated = [detect_other_langs(r['comments']) if int(r['num_of_sents']) > 1 else (0,None) for i, r in table.iterrows()]
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

def pipeline(reviews, stopwords=[], vocab=None):
    reviews_encoded = {}
    vocab_counts = {}
    vocab_doc_count = {}

    is_ext_vocab = True
    if vocab is None:
        is_ext_vocab = False
        vocab = {'<OOV>': 0}
    for key in reviews.keys():
        r = []
        tokenized_r = utils.tokenize(reviews[key].lower())
        for token in tokenized_r:
            #token = ps.stem(token)
            if token in stopwords:
                continue
            if not is_ext_vocab and token not in vocab:
                vocab[token] = len(vocab)
                vocab_counts[token] = 1
            if token not in vocab:
                token_id = vocab['<OOV>']
                vocab_counts['<OOV>'] += 1
            else:
                token_id = vocab[token]
                vocab_counts[token] += 1
            r.append(token_id)
        reviews_encoded[key] = r
    idf_dict = []
    for key in vocab.keys():
        if key in vocab_doc_count.keys():
            idf_dict.append(np.log10(N/vocab_doc_count[key]))
        else:
            idf_dict.append(0)
    return reviews_encoded, vocab_counts, vocab, idf_dict

def create_vocab_count_table(counts):
    return pd.DataFrame({'counts':[counts[k] for k in counts.keys()], 'word':[k for k in counts.keys()]})
