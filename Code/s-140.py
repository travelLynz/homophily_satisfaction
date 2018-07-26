def save_review_sentence_short(revset, name, path=""):
    din = os.path.join(path,'s140','in', name  + '.txt')
    file = open(din, 'w')
    for i, c in zip(revset.id, revset.comments):
        sents = sent_tokenize(c)
        for n, sent in zip(range(len(sents)), sents):
            file.write(str(i) + "-" + str(n) + " : " + sent + "\r\n")
    file.close()
