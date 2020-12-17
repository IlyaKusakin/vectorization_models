# -*- coding: utf-8 -*-
"""
Module with unit-tests for project

@author: Kusakin Ilya
"""

from source.word2vec import Word2Vec 
from source.tfidf import TfIdfVectorizer 

stopwords_file = 'data\\ru_stopwords.txt'
    
def test_read_corpora(load_corpora):
    corpora = load_corpora
    assert len(corpora) > 0
    
def test_delete_stopwords(load_corpora):
    corpora = load_corpora
    corpora.delete_stopwords()
    stopword = open(stopwords_file, 'r', encoding='UTF-8').readlines()[-1].rstrip()
    
    for text_dict in corpora:
        assert stopword not in text_dict["text_data"]

def test_clean_texts(load_corpora):
    corpora = load_corpora
    corpora.clean_texts()
    
    for text_dict in corpora:
        assert '.' not in text_dict["text_data"]
        assert '1' not in text_dict["text_data"]
        
def test_vectorizer_dicts(load_corpora):
    corpora = load_corpora 
    word2vec = Word2Vec(corpora)
    tfidf = TfIdfVectorizer(corpora)
    
    assert word2vec.get_word2ind()["<UNK>"] == 0
    assert tfidf.get_word2ind()["<PAD>"] == 1
    assert len(word2vec.get_ind2word()) == len(tfidf.get_ind2word())

    