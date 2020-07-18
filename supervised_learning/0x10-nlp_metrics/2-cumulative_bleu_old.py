#!/u1sr/bin/env python3
""" N-gram BLEU score """
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from nltk import ngrams
from collections import Counter


def grams(sentence, n):

    new = []
    ln = len(sentence)
    for i, word in enumerate(sentence):
        s = word
        counter = 0
        for j in range(1, n):
            if ln > i+j:
                s += " " + sentence[i+j]
                counter += 1
        if counter == j:
            new.append(s)
    return new

def transform_grams(references, sentence, n):
    if n == 1:
        return references, sentence
    new_sentence = grams(sentence, n)
    new_ref = []
    for ref in references:
        new_r = grams(ref, n)
        new_ref.append(new_r)

    return new_ref, new_sentence


def calc_precision(references, sentence, i):

    references, sentence = transform_grams(references, sentence, i)
    print("ref", references)
    print("sent", sentence)

    c = len(sentence)
    r_list = np.array([np.abs(len(s)-c) for s in references])
    r_ind = np.argwhere(r_list == np.min(r_list))
    lens = np.array([len(s) for s in references])[r_ind]
    r = np.min(lens)

    sentence_dict = {x:sentence.count(x) for x in sentence}

    shown = {}
    rep = []
    for ref in references:
        ref_dict = {i:ref.count(i) for i in ref}
        rep.append(ref_dict)
        for gram in sentence_dict.keys():
            if not gram in shown.keys():
                shown[gram] = 0
            if gram in ref_dict.keys():
                shown[gram] = sentence_dict[gram]
    ceiling = {}
    for ref in rep:
        for gram in ref:
            if not gram in ceiling.keys() or ceiling[gram] < ref[gram]:
                ceiling[gram] = ref[gram]


    # Clipped
    print("shown", shown)
    print("rep", rep)
    print("ceiling", ceiling)
    for gram in shown.keys():
        if shown[gram] and shown[gram] > ceiling[gram]:
            shown[gram] = ceiling[gram]

    print("c:", c, "r", r)
    precision = sum(shown.values()) / c
    return precision, c, r

def ngram_bleu(references, sentence, n):
    """ calculates the n-gram BLEU score for a sentence:

        - references is a list of reference translations
            each reference translation is a list of the words
            in the translation
        - sentence is a list containing the model proposed sentence
        - n is the size of the n-gram to use for evaluation

    Returns: the unigram BLEU score
    """
    c = len(sentence)
    r_list = np.array([np.abs(len(s)-c) for s in references])
    r_ind = np.argwhere(r_list == np.min(r_list))
    lens = np.array([len(s) for s in references])[r_ind]
    r = np.min(lens)

    precisions = np.empty((n,))
    for i in range(1, n+1):
        precisions[i-1], _, _ = calc_precision(references, sentence, i)
    if c > r:
        BP = 1
    else:
        BP = np.exp(1-(r/c))
    print("BP", BP)
    print(precisions)
    w = np.empty((n, ))
    w[:] = 1/n
    print("w", w)
    Bleu = BP * np.sum(np.log(precisions) * w)
    print("Bleu: ", Bleu)
    score = sentence_bleu(references, sentence, weights=(0, 1, 0, 0))
    print("nltk result", score)
    return Bleu
