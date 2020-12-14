# -*- coding: utf-8 -*-
"""
Main module of programm for words vectorizing.
 
@author: Kusakin Ilya
"""
import numpy as np

from source.corpora import Corpora
from source.word2vec import Word2Vec
from source.tfidf import TfIdfVectorizer


def cosine_similarity(vec1,vec2):
    """Function for calcilating cosine similarity between two numpy vector-arrays."""
    num = abs(vec1@vec2) + 1e-2
    return 1 - num/(np.sum(vec1**2) * np.sum(vec2**2) + 1e-2)


if __name__ == '__main__':
    
    print("Hello, this is a program for words vectorization, type a filename with dataset.")
        
    filename = input("File with dataset: ")
    
    print("Preprocessing of texts...")
    corpora = Corpora(filename)
    corpora.lemmatize_texts()
    corpora.clean_texts()
    corpora.delete_stopwords()
    
    print('-' * 60)
    print("Choose vectorizer model.")
    vec_type = ''
    while vec_type != 'Word2Vec' and vec_type != 'TfIdf':
        vec_type = input("Type 'Word2Vec' or 'TfIdf': ")
    
    if vec_type == "Word2Vec":
        
        print("Word2Vec model initialization...")
        w2v = Word2Vec(corpora)
        
        print("Enter 2 word for calculating cosine distance between them.")
        word_1 = input("Word #1: ")
        word_2 = input("Word #2: ")
        
        print('-' * 60)
        
        print("Fit all model or only two neccessary words?")
        
        fit_all = ''
        while fit_all != 'all' and fit_all != "two": 
            fit_all = input("Type 'all' or 'two': ")
        
        if fit_all == 'all':
            
            print("Word2Vec model is fitting, please wait...")      
            w2v.fit(speed=15)
            
            vec_1 = w2v[word_1]
            vec_2 = w2v[word_2]
            print("cossim = ", cosine_similarity(vec_1, vec_2))
            
            
            print('-' * 60)
            print("Downloading vectors to pickle file...")
            w2v.download_to_pickle()
            print("Saving vectors to json file...")
            w2v.download_to_json()
        
        else:
            
            vec_1 = w2v.get_fit_vector(word_1)
            vec_2 = w2v.get_fit_vector(word_2)
            print("cossim = ", cosine_similarity(vec_1, vec_2))
        
    if vec_type == "TfIdf":
        
        print("TfIdf model initialization...")
        tfidf = TfIdfVectorizer(corpora)
        
        print("Calculating document frequency...")
        tfidf.calc_df()
        print("Calculating turn frequency...")
        tfidf.calc_tf()
        
        print("Creating dictionary of vectors...")
        tfidf.create_word2vec()
        
        print('-' * 60)
        
        print("Enter 2 word for calculating cosine distance between them.")
        word_1 = input("Word #1: ")
        word_2 = input("Word #2: ")
        
        vec_1 = tfidf[word_1]
        vec_2 = tfidf[word_2]
        print("cossim = ", cosine_similarity(vec_1, vec_2))
        
        print('-' * 60)
        
        print("Downloading vectors to pickle file...")
        #tfidf.download_to_pickle()
        print("Saving vectors to json file...")
        #tfidf.download_to_json()
    
    print('End.')