#!/u1sr/bin/env python3
""" Unigram BLEU score """
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from nltk import ngrams
from collections import Counter


def uni_bleu(references, sentence):
    """ calculates the unigram BLEU score for a sentence:

        - references is a list of reference translations
            each reference translation is a list of the words
            in the translation
        - sentence is a list containing the model proposed sentence

    Returns: the unigram BLEU score
    """
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
    if c > r:
        BP = 1
    else:
        BP = np.exp(1-(r/c))
    print("BP", BP)
    Bleu = BP * precision
    print("Bleu: ", Bleu)
    score = sentence_bleu(references, sentence, weights=(1, 0, 0, 0))
    print("nltk result", score)
    return Bleu
