# -*- coding: utf-8 -*-
"""
Source module with TfIdfVectorizer class

@author: Kusakin Ilya
"""

import numpy as np
from scipy.sparse import lil_matrix
import re
from threading import Thread
import json
import pickle

from .vectorizer import _Vectorizer     
from .loggers.declogger import FunctionLogger
logger = FunctionLogger(__name__)

class TfIdfVectorizer(_Vectorizer):
    """
    
    Child class of Vectorizer.
    
    Vectorize words from corpora by Tf-Idf algorythm.
    
    """
    @logger
    def __init__(self, corpora, vocab_size=1000, vector_size=300):
        super().__init__(corpora, vocab_size)
        
        self.__name__ = 'TfIdf'
        self._vector_size = vector_size
        self._tf = lil_matrix((len(self._words), len(self._corpora)))
        self._df = np.zeros(len(self._words))
        
    @logger
    def calc_tf(self):
        """Calculates turn frequency of words from wordset"""
        for i in range(len(self._words)):
            th = Thread(target=self.calc_tf_thread, args=(i,) )
            th.start()
    
    def calc_tf_thread(self, i):
        """Support method for base method that used threads."""
        word = self._ind2word[i]
        self._tf[i] = lil_matrix([self.count_substring(text_dict["text_data"], word) for text_dict in self._corpora])
      
    @logger           
    def calc_df(self):
        """Calculates document frequency of words from wordset"""
        for i in range(len(self._words)):
            th = Thread(target=self.calc_df_thread, args=(i,) )
            th.start()
    
    def calc_df_thread(self,i):
        """Support method for base method that used threads."""
        word = self._ind2word[i]
        self._df[i] =  sum(map(lambda x: word in x["text_data"], self._corpora))
 
    @logger
    def create_word2vec(self):
        """Creates word2vec dictionary via calculated tf and df."""
        for word,ind in self._word2ind.items(): 
            tf = self._tf[ind].toarray()[0]
            df = self._df[ind]
            tf_idf = tf*np.log(len(self._corpora)/(df+1)) 
            self._word2vec[word] = tf_idf
      
    def count_substring(self, string, substring):
        """Method for calculating number of substring in string."""
        return len(re.findall(substring, string))        
     
    def get_tf(self):
        return self._tf
    
    def get_df(self):
        return self._df
    
    def __getitem__(self,word):  
        
        if word in self._word2vec:
            return self._word2vec[word]
        
        else:
            return self._word2vec["<UNK>"]
    
    @logger
    def download_to_json(self, filename='tfidf_vectors.json'):
        
            listed_word2vec = {k: v.tolist() for k,v in self._word2vec.items()} 
            with open(filename, 'w') as fp:
                json.dump(listed_word2vec, fp, ensure_ascii=False)
        
    @logger
    def upload_from_json(self, filename='tfidf_vectors.json'):
        
        try:  
            with open(filename, 'r') as fp:
                listed_word2vec = json.load(fp)
                self._word2vec = {k: np.array(v) for k,v in listed_word2vec.items()}
        
        except:
            print("Error: wrong json filename.")
        
    @logger
    def download_to_pickle(self, filename='tfidf_vectors.p'):
    
        with open(filename, 'wb') as fp:
            pickle.dump(self._word2vec, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    @logger
    def upload_from_pickle(self, filename='tfidf_vectors.p'):
        
        try:
            with open(filename , 'rb') as fp:
                self._word2vec = pickle.load(fp)       
        except:
            print("Error: wrong pickle filename.")