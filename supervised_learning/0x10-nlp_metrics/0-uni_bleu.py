#!/u1sr/bin/env python3
""" Unigram BLEU score """
import numpy as np


def uni_bleu(references, sentence):
    """ calculates the unigram BLEU score for a sentence:

        - references is a list of reference translations
            each reference translation is a list of the words
            in the translation
        - sentence is a list containing the model proposed sentence

    Returns: the unigram BLEU score
    """
    # calculates c and r
    c = len(sentence)
    r_list = np.array([np.abs(len(s)-c) for s in references])
    r_ind = np.argwhere(r_list == np.min(r_list))
    lens = np.array([len(s) for s in references])[r_ind]
    r = np.min(lens)

    # sentence as a dictionary
    sentence_dict = {x: sentence.count(x) for x in sentence}

    # couts if the gram is shown
    shown = {}
    rep = []
    for ref in references:
        ref_dict = {i: ref.count(i) for i in ref}
        rep.append(ref_dict)
        for gram in sentence_dict.keys():
            if gram not in shown.keys():
                shown[gram] = 0
            if gram in ref_dict.keys():
                shown[gram] = sentence_dict[gram]

    # creates ceiling to be clipped
    ceiling = {}
    for ref in rep:
        for gram in ref:
            if gram not in ceiling.keys() or ceiling[gram] < ref[gram]:
                ceiling[gram] = ref[gram]

    # Clipping
    for gram in shown.keys():
        if shown[gram] and shown[gram] > ceiling[gram]:
            shown[gram] = ceiling[gram]

    precision = sum(shown.values()) / c
    if c > r:
        BP = 1
    else:
        BP = np.exp(1-(r/c))
    Bleu = BP * precision
    return Bleu
