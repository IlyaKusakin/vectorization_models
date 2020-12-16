# -*- coding: utf-8 -*-
"""
Source module with Word2Vec class

@author: Kusakin Ilya
"""

import numpy as np
import nltk
import razdel
from threading import Thread
import json
import pickle

from .vectorizer import _Vectorizer
from .loggers.declogger import FunctionLogger
logger = FunctionLogger(__name__)     

class Word2Vec(_Vectorizer):
    """
    
    Child class of Vectorizer.
    
    Vectorize words from corpora by word2vec mechanism.
    
    """
    @logger
    def __init__(self, corpora, vocab_size=1000, vector_size=300, window_size=5 ):
        super().__init__(corpora, vocab_size)
        
        self.__name__ = 'w2v'
        self._vector_size = vector_size
        self._window_size = window_size if window_size>3 else 3
        
        self._context_dict = {ind:[] for word, ind in self._word2ind.items()}
        self._V = np.random.randn(len(self._words), vector_size)
        self._U = np.random.randn(vector_size, len(self._words))
        
        self._tokenizer = nltk.WordPunctTokenizer()
        self.calc_context()
        
    @logger
    def calc_context(self, window_size = 5):
        """
        
        Calculates neighbours of each word with fixed window size. 
        Creates context dict with indexes of center and context words.
        
        """
        self._window_size = window_size
        
        for text_dict in self._corpora:  
            sentences = razdel.sentenize(text_dict["text_data"])
            
            for sent in sentences:   
                tokens = [token for token in self._tokenizer.tokenize(sent.text) if token.isalpha()]
                idx_tokens =  [self._word2ind[token] 
                               if token in self._word2ind else self._word2ind["<UNK>"] for token in tokens]

                for side_idx in range(self._window_size//2):  
                    left_side = idx_tokens[side_idx+1 : (self._window_size//2)+side_idx+1]
                    right_side = idx_tokens[-(self._window_size//2)-side_idx-1 : -side_idx-1]
                    self._context_dict[idx_tokens[side_idx]].extend(left_side)
                    self._context_dict[idx_tokens[-side_idx-1]].extend(right_side)


                if len(idx_tokens) > self._window_size-1:
                    for i in range(self._window_size//2, len(idx_tokens) - self._window_size//2):   
                        for step in range(1, self._window_size//2 +1):
                            self._context_dict[idx_tokens[i]].append(idx_tokens[i+step])
                            self._context_dict[idx_tokens[i]].append(idx_tokens[i-step]) 
        
        del self._context_dict[0] #element for <UNK> token
        del self._context_dict[1] #element for <PAD> token 
        
    @logger
    def fit(self, lr=1e-2, speed=10):
        """Method for fitting vectors via word2vec algorythm."""
        for center_token, context in self._context_dict.items():     
            th = Thread(target=self.fit_vector, args=(center_token,context[:10], lr) )
            th.start()   
        self.create_word2vec()     
            
    @logger
    def create_word2vec(self):
        """Fills word2vec dictionary with vectors from V matrix."""
        for word, idx in self._word2ind.items():
            self._word2vec[word] = self._V[idx]
    
    @logger
    def fit_vector(self, center_token, context_tokens, lr):
        """Method for fitting one word."""
        for context_token in context_tokens:      
            prob_numerator = np.exp(self._V[center_token] @ self._U)
            prob_denominator = np.sum(prob_numerator)

            grad = self._U[:, context_token] - np.sum(prob_numerator * self._U / prob_denominator)
            self._V[center_token] -= lr * grad / len(context_tokens)
        
    def get_fit_vector(self, word, speed=1000, lr=1):
        """
        
        Fits and returns vector for word.
        
        Speed argument defines number of neighbours that needs for fitting.
        
        """
        if word not in self._word2ind:
            return np.zeros(self._vector_size)
        
        center_token = self._word2ind[word]
        context_tokens = self._context_dict[center_token][:speed]

        for context_token in context_tokens:       
            prob_numerator = np.exp(self._V[center_token] @ self._U)
            prob_denominator = np.sum(prob_numerator) + 1e-2

            grad = self._U[:, context_token] - np.sum(prob_numerator * self._U / prob_denominator)
            self._V[center_token] -= lr * grad / len(context_tokens)
            
        return self._V[center_token]
    
    def __getitem__(self, word):
        
        if word in self._word2vec:
            return self._word2vec[word]
        
        else:
            return self._word2vec["<UNK>"]
        
    @logger
    def download_to_json(self, filename='w2v_vectors.json'):
        
        listed_word2vec = {k: v.tolist() for k,v in self._word2vec.items()} 
        with open(filename, 'w') as fp:
            json.dump(listed_word2vec, fp, ensure_ascii=False)
    
    @logger
    def upload_from_json(self, filename='w2v_vectors.json'):
        
        try:
            with open(filename, 'r') as fp:
                listed_word2vec = json.load(fp)
                self._word2vec = {k: np.array(v) for k,v in listed_word2vec.items()}
        
        except:
            print("Error: wrong json filename.")
    
    @logger
    def download_to_pickle(self, filename='w2v_vectors.p'):
    
        with open(filename, 'wb') as fp:
            pickle.dump(self._word2vec, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    @logger
    def upload_from_pickle(self, filename='w2v_vectors.p'):
            
        try:
            with open(filename , 'rb') as fp:
                self._word2vec = pickle.load(fp)
        
        except:
            print("Error: wrong pickle filename.")