# -*- coding: utf-8 -*-
"""
Main module of programm for words vectorizing.
 
@author: Kusakin Ilya
"""
import numpy as np

def cosine_distance(vec1,vec2):
    num = abs(vec1@vec2)
    return num/(np.sum(vec1**2) * np.sum(vec2**2))

from source.corpora import Corpora
from source.Vectorizers import TfIdfVectorizer, Word2Vec

if __name__ == '__main__':
    
    print("Hello, this is a program for word vectorizing, type a filename with dataset.")
        
    filename = input("File with dataset: ")
    
    print("Preprocessing of texts...")
    corpora = Corpora(filename)
    corpora.lemmatize_texts()
    corpora.clean_texts()
    corpora.delete_stopwords()
    
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
        
        print("Fit all model or only two neccessary words?")
        
        fit_all = ''
        while fit_all != 'all' and fit_all != "two": 
            fit_all = input("Type 'all' or 'two': ")
        
        if fit_all == 'all':
            
            print("Word2Vec model is fitting, please wait...")      
            w2v.fit(speed=15)
            
            vec_1 = w2v[word_1]
            vec_2 = w2v[word_2]
            print("cosdistance = ", round(cosine_distance(vec_1, vec_2), 8))
            
            print("Downloading vectors to pickle file...")
            w2v.download_to_pickle()
            print("Saving vectors to json file...")
            w2v.download_to_json()
        
        else:
            
            vec_1 = w2v.get_fit_vector(word_1)
            vec_2 = w2v.get_fit_vector(word_2)
            print("cosdistance = ", round(cosine_distance(vec_1, vec_2), 8))
        
    if vec_type == "TfIdf":
        
        print("TfIdf model initialization...")
        tfidf = TfIdfVectorizer(corpora)
        
        print("Calculaing document frequency...")
        tfidf.calc_df()
        print("Calculaing turn frequency...")
        tfidf.calc_tf()
        
        print("Creating dictionary of vectors...")
        tfidf.create_word2vec()
        
        
        print("Enter 2 word for calculating cosine distance between them.")
        word_1 = input("Word #1: ")
        word_2 = input("Word #2: ")
        
        vec_1 = tfidf[word_1]
        vec_2 = tfidf[word_2]
        print("cosdistance = ", cosine_distance(vec_1, vec_2))
        
        print("Downloading vectors to pickle file...")
        tfidf.download_to_pickle()
        print("Saving vectors to json file...")
        tfidf.download_to_json()
    
    print('End.')
    
    
