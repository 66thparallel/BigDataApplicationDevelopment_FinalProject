# coding: utf-8
# !/usr/bin/python3
"""
Authors: Yu-Ting Chiu, Jane Liu
Classes:
    Tokenizer:
        Accepts a list of words and outputs tokenized text.
    RemoveStopWords:
        Accepts a list of tokens and removes stop words.
    Unigrams:
        Returns the top 100 unigrams and prints to the console and ngrams.txt.
    Preprocessor:
        Calls Tokenizer, RemoveStopWords, and lemmatizes the text. Finds the top 100 unigrams.
"""

import re
import string
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams


# Get the top 100 unigrams
class Tokenizer:
    def __init__(self, text):
        self._text = text
        self._tokenized_text = []

    def tokenize(self):
        # Remove punctuation, empty elements, numbers, and dates
        pattern = re.compile('[0-9]+')
        self._tokenized_text = [''.join(c for c in s if c not in string.punctuation) for s in self._text]
        self._tokenized_text[:] = [word for word in self._tokenized_text if not pattern.match(word) and word != '']

        return self._tokenized_text


class RemoveStopWords:
    def __init__(self, text):
        self._text = text
        self._stopwords = []

    def removestopwords(self):
        with open('/scratch/jl860/bdad/fp/preprocess/stopwords.txt', 'r') as g:
            self._stopwords = g.read().splitlines()
        for word in self._stopwords:
            self._text = [value for value in self._text if value.lower() != word]

        return self._text


class Unigrams:
    def __init__(self, topics):
        self._topics = topics
        self._unigrams = []

    def get_top_unigrams(self):

        # Find the most frequently occuring unigrams
        word_freq = Counter(self._topics)

        common_words = word_freq.most_common(100)
        dict_unigrams = dict(common_words)

        # convert dict to list for convenience
        [self._unigrams.append(x) for x in dict_unigrams]

        return self._unigrams


class Preprocessor:

    def __init__(self):
        self._reviews = []
        self._cleantext = []
        self._temptext = []
        self._preprocessedlist = []
        self._unigrams = []
        self._bigrams = []
        self._ngrams = []

    def preprocess(self, rdd):
        self._reviews = rdd.read().split()

        # Tokenize the text file
        self._temptext = Tokenizer(self._reviews)
        self._cleantext = self._temptext.tokenize()

        # Remove stop words
        self._temptext = RemoveStopWords(self._cleantext)
        self._cleantext = self._temptext.removestopwords()

        # Lemmatize the text
        lemma_text = []
        lemmatizer = WordNetLemmatizer()

        for word in list(self._cleantext):
            new_word = lemmatizer.lemmatize(word)
            lemma_text.append(new_word)

        for word in lemma_text:
            self._preprocessedlist.append(word)

        unitemp = Unigrams(self._preprocessedlist)
        self._unigrams = unitemp.get_top_unigrams()

        [self._ngrams.append(x) for x in self._unigrams]

        return self._ngrams
