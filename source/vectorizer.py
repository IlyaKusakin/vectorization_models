# -*- coding: utf-8 -*-
"""
Module with abstract Vectorizer class

@author: Kusakin Ilya
"""

import numpy as np
import nltk
from abc import abstractmethod

class _Vectorizer():
    """
    
    Abstract class of vectorizer model.
    
    Contains methods for creating set of words from corpora
    and dict word2ind/ind2word.
    
    """
    def __init__(self, corpora, vocab_size = 1000, tokenizer = nltk.WordPunctTokenizer()):
    
        self._corpora = corpora
        self._tokenizer = tokenizer
        self._vocab_size = vocab_size
        self.corpora_preprocessing()
        
        self._words = set()
        self._word2vec = {"<UNK>":np.zeros(vocab_size), "<PAD>":np.zeros(vocab_size)}
        self._word2ind = {"<UNK>":0, "<PAD>":1}
        self._ind2word = {0:"<UNK>", 1:"<PAD>"}
        self._special_tokens = {"<UNK>", "<PAD>"}
        
        self.create_wordset()
        self.create_dicts()
        
    def corpora_preprocessing(self):
        """Method for corpora cleaning from rubbish."""
        self._corpora.lemmatize_texts()
        self._corpora.clean_texts()
        self._corpora.delete_stopwords()
        
    def create_wordset(self):
        """
        Method for creating word's set from all corpora texts.
        
        Max size of set == self.vocab_size
        
        """
        all_words = dict()
        
        for text in self._corpora:
            for token in self._tokenizer.tokenize(text["text_data"]):
                if token in all_words:
                    all_words[token]+=1
                else:
                    all_words[token]=1
                    
        words_list = list(all_words.items())
        words_list.sort(key=lambda i: i[1], reverse=True)
        self._words = {elem[0] for elem in words_list[:self._vocab_size]} | self._special_tokens
            
    def create_dicts(self):
        """Create word2ind, ind2word dicts via wordset."""
        for ind, word in enumerate(self._words - self._special_tokens):
            self._word2ind[word] = ind+len(self._special_tokens)
            self._ind2word[ind+len(self._special_tokens)] = word
            
    def get_word_by_ind(self, ind):
        """Getting word from ind2word dictionary by 'ind' argument."""
        if ind in self._ind2word:
            return self._ind2word[ind]
        
        else:
            return self._ind2word[0]
    
    def get_ind_by_word(self, word):
        """Getting ind from word2ind dictionary by 'word' argument."""
        if word in self._word2ind:
            return self._word2ind[word]
        
        else:
            return self._word2ind['<UNK>']
    
    def get_word2ind(self):
        return self._word2ind
    
    def get_ind2word(self):
        return self._ind2word
    
    def get_word2vec(self):
        return self._word2vec
    
    @abstractmethod
    def download_to_json(self, filename='vectors.json'):
        """Method for saving word2vec dictionary into json file."""
        pass 
       
    @abstractmethod
    def upload_from_json(self, filename='vectors.json'):
        """Method for uploading word2vec dictionary from json file."""
        pass
    
    @abstractmethod
    def download_to_pickle(self, filename='vectors.p'):
        """Method for serializing word2vec dictionary via pickle."""
        pass
    
    @abstractmethod
    def upload_from_pickle(self, filename='vectors.p'):
        """Method for uploading word2vec dictionary from serialized pickle file."""
        pass
    
    @abstractmethod
    def __getitem__(self, word):
        """Return vector from word2vec dictionary via 'word' argument."""
        pass