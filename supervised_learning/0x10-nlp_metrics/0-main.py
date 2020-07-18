#!/usr/bin/env python3

uni_bleu = __import__('0-uni_bleu').uni_bleu

references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
# references = [["the", "cat", "on", "mat"], ["there", "is", "a", "cat", "on", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]
# sentence = ["the", "cat", "the", "cat", "on", "the", "mat"]
# sentence = ["the", "the", "the", "the", "the", "the", "mat"]
# sentence = ["the", "the", "the", "the", "the", "the", "mat"]


print(uni_bleu(references, sentence))
