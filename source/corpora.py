# -*- coding: utf-8 -*-
"""
Source module with Corpora class.

@author: Kusakin Ilya
"""

import nltk
import pymorphy2
import string
import pandas as pd

from .loggers.declogger import FunctionLogger
logger = FunctionLogger(__name__)

class Corpora():
    """
    
    Class for creating corpora from file with dataset.
    
    Consist function for texts reading and preprocessing.
    Also have methods for iterating and getting item.
    
    """
    @logger
    def __init__(self, corpora_file, file_type='csv', tokenizer=nltk.WordPunctTokenizer(), 
                 column = 'annotation', lemmatizer = pymorphy2.MorphAnalyzer(), lang="ru"):
        
        self._lang = lang
        self._texts = []
        self._corpora_file = corpora_file
        self._tokenizer = tokenizer
        self._lemmatizer = lemmatizer
        self._file_type = file_type
        self._text_column = column
        
        self.read_corpora()
    
    @logger
    def read_corpora(self):
        """Function for reading file with texts corpora"""
        
        if self._file_type == 'csv':
            
            try:     
                data = pd.read_csv(self._corpora_file, sep="|")
                texts = data[self._text_column]
                for text in texts:
                    self._texts.append({"text_data":text, "meta":[]})
            
            except:     
                print("Error: Wrong filename. Check the path to file with texts.")
                
    @logger            
    def clean_texts(self):
        """Cleans texts from punctuation and numerics"""
        
        for text_dict in self._texts:
            text = text_dict["text_data"].lower()
            
            for sym in string.punctuation+'0123456789':
                if sym in text:
                    text = text.replace(sym, "")
            
            text_dict["text_data"] = text
    
    @logger
    def lemmatize_texts(self, fast=True):
        """
        
        Function for corpora lemmatization.
        
        Fast argument activates fast lemmatizaion with low quality.
        For perfect lemmatization set "fast" to False value.
        
        """
        for text_dict in self._texts:
            text = text_dict["text_data"]

            if fast:
                lemmatized_text = self._lemmatizer.parse(text)[0].normal_form
            
            else:
                tokenized_text = self._tokenizer.tokenize(text)
                lemmatized_text = " ".join([self._lemmatizer.parse(token)[0].normal_form for token in tokenized_text if token.isalpha()])                 
            
            text_dict["text_data"] = lemmatized_text
    
    @logger
    def delete_stopwords(self, stopwords_file='ru_stopwords.txt' ):
        """
        
        Function for deleting stopwords from texts in corpora.
        
        You can set your own "stopwords_file". 
        
        """
        stopwords = open(stopwords_file, 'r', encoding='UTF-8').readlines()
        for text_dict in self._texts:      
            tokenized_text = self._tokenizer.tokenize(text_dict["text_data"])
            tokenized_text = [token for token in tokenized_text if token+'\n' not in stopwords]
            text_dict["text_data"] = " ".join(tokenized_text)
            
    @logger        
    def set_tokenizer(self, tokenizer):
        """Setter for tokenizer. Default is nltk.WordPunctTokenizer()."""
        self._tokenizer = tokenizer
    
    def get_texts(self):
        """Getter for texts dataset."""
        return self._texts
    
    def __len__(self):
        return len(self._texts)
    
    def __getitem__(self, item):
        return self._texts[item]
    
    def __iter__(self):
        for text in self._texts:
            yield text