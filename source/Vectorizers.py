# -*- coding: utf-8 -*-
"""
Source module with Vectorizer-family classes

@author: Kusakin Ilya
"""

from .corpora import Corpora 

import numpy as np
from scipy.sparse import lil_matrix
import nltk
import re
import razdel
from abc import abstractmethod
from threading import Thread
import json
import pickle
    

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
    
    
class TfIdfVectorizer(_Vectorizer):
    """
    
    Child class of Vectorizer.
    
    Vectorize words from corpora by Tf-Idf algorythm.
    
    """
    def __init__(self, corpora, vector_size=300):
        super().__init__(corpora)
        
        self.__name__ = 'TfIdf'
        self._vector_size = vector_size
        self._tf = lil_matrix((len(self._words), len(self._corpora)))
        self._df = np.zeros(len(self._words))
        

    def calc_tf(self):
        """Calculates turn frequency of words from wordset"""
        for i in range(len(self._words)):
            th = Thread(target=self.calc_tf_thread, args=(i,) )
            th.start()
                     
    def calc_tf_thread(self, i):
        """Support method for base method that used threads."""
        word = self._ind2word[i]
        self._tf[i] = lil_matrix([self.count_substring(text_dict["text_data"], word) for text_dict in self._corpora])
                 
    def calc_df(self):
        """Calculates document frequency of words from wordset"""
        for i in range(len(self._words)):
            th = Thread(target=self.calc_df_thread, args=(i,) )
            th.start()
                
    def calc_df_thread(self,i):
        """Support method for base method that used threads."""
        word = self._ind2word[i]
        self._df[i] =  sum(map(lambda x: word in x["text_data"], self._corpora))
    
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
    
    def download_to_json(self, filename='tfidf_vectors.json'):
        
            listed_word2vec = {k: v.tolist() for k,v in self._word2vec.items()} 
            with open(filename, 'w') as fp:
                json.dump(listed_word2vec, fp)
        
    def upload_from_json(self, filename='tfidf_vectors.json'):
        
        try:  
            with open(filename, 'r') as fp:
                listed_word2vec = json.load(fp)
                self._word2vec = {k: np.array(v) for k,v in listed_word2vec.items()}
        
        except:
            print("Error: wrong json filename.")
        
    def download_to_pickle(self, filename='tfidf_vectors.p'):
    
        with open(filename, 'wb') as fp:
            pickle.dump(self._word2vec, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    def upload_from_pickle(self, filename='tfidf_vectors.p'):
        
        try:
            with open(filename , 'rb') as fp:
                self._word2vec = pickle.load(fp)       
        
        except:
            print("Error: wrong pickle filename.")

class Word2Vec(_Vectorizer):
    """
    
    Child class of Vectorizer.
    
    Vectorize words from corpora by word2vec mechanism.
    
    """
    def __init__(self, corpora, vector_size=300, window_size=5 ):
        super().__init__(corpora)
        
        self.__name__ = 'w2v'
        self._vector_size = vector_size
        self._window_size = window_size if window_size>3 else 3
        
        self._context_dict = {ind:[] for word, ind in self._word2ind.items()}
        self._V = np.random.randn(len(self._words), vector_size)
        self._U = np.random.randn(vector_size, len(self._words))
        
        self._tokenizer = nltk.WordPunctTokenizer()
        
        self.calc_context()
        
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
        
    def fit(self, lr=1e-2, speed=10):
        """Method for fitting vectors via word2vec algorythm."""
        for center_token, context in self._context_dict.items():
            
            th = Thread(target=self.fit_vector, args=(center_token,context[:10], lr) )
            th.start()
            
        self.create_word2vec()     
            
    def create_word2vec(self):
        """Fills word2vec dictionary with vectors from V matrix."""
        for word, idx in self._word2ind.items():
            self._word2vec[word] = self._V[idx]
    
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
        
    def download_to_json(self, filename='w2v_vectors.json'):
        
        listed_word2vec = {k: v.tolist() for k,v in self._word2vec.items()} 
        with open(filename, 'w') as fp:
            json.dump(listed_word2vec, fp)
    
    def upload_from_json(self, filename='w2v_vectors.json'):
        
        try:
            with open(filename, 'r') as fp:
                listed_word2vec = json.load(fp)
                self._word2vec = {k: np.array(v) for k,v in listed_word2vec.items()}
        
        except:
            print("Error: wrong json filename.")
    
    def download_to_pickle(self, filename='w2v_vectors.p'):
    
        with open(filename, 'wb') as fp:
            pickle.dump(self._word2vec, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    def upload_from_pickle(self, filename='w2v_vectors.p'):
            
        try:
            with open(filename , 'rb') as fp:
                self._word2vec = pickle.load(fp)
        
        except:
            print("Error: wrong pickle filename.")