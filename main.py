# -*- coding: utf-8 -*-
"""
Main module of console application.

@author: Kusakin Ilya
"""

import argparse
from source.corpora import Corpora
from source.word2vec import Word2Vec
from source.tfidf import TfIdfVectorizer

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=""" This is the console
                                     application for words vectorization.""")
    
    parser.add_argument("--corpora", default="data\\corpora.csv", type=str, 
                        help="Name of input csv-file with corpora of texts.")
    
    pasrser.add_argument("--column", default="annotation", type=str,
                         help="""Name of column in csv-file with texts, 
                         default is 'annotaion' for 'corpora.csv' file.""")

    parser.add_argument("--output_file", default="vectors.json", type=str,
                        help="Name of output json-file with values of word-vectors.")
    
    parser.add_argument("--model", choices=["Word2Vec", "TFIDF"],
                        required=True, type=str, help="Type of vectorizer mofel.")

    parser.add_argument("--vocab_size", default=1000, type=int, 
                        help="""Number of words that will be included into vocabulary, 
                        1000 is default.""")
    
    parser.add_argument("--vector_size", default=100, type=int, 
                        help="""Size of vectors that will be calculated via Word2Vec model, 
                        100 is default.""")
    
    parser.add_argument("--window", default=5, type=int, 
                        help="Window size for Word2Vec algorythm, 5 is default.")
    
    parser.add_argument("--lr", default=1e-1, type=float, 
                        help="""Learning rate for backpropagation algorythm in Word2Vec model, 
                        1 is default""")
    
    parser.add_argument("--speed", default=10, type=int, 
                        help="""Number of word's neighbours that calculates in Word2Vec.
                        Directly impacts on computing time, 10 is default.""")
    
    args = parser.parse_args()
    input_filename = args.corpora
    model = args.model
    output_filename = args.output_file 
    vector_size = args.vector_size
    vocab_size = args.vocab_size
    window = args.window
    lr = args.lr
    speed = args.speed
    
    print("Data preprocessing...")
    corpora = Corpora(input_filename)
    corpora.lemmatize_texts()
    corpora.clean_texts()
    corpora.delete_stopwords()
    
    if model == "Word2Vec":
        print("Word2Vec model initialization...")
        w2v = Word2Vec(corpora, vector_size=vector_size, vocab_size=vocab_size, window_size=window)
        print("Word2Vec is fitting...")
        w2v.fit(lr,speed)
            
        w2v.download_to_json(output_filename)
        print("Vectors were saved to " + output_filename)
        
    else:
        print("TfIdf model initialization...")
        tfidf = TfIdfVectorizer(corpora, vocab_size)
        
        print("Calculating document frequency...")
        tfidf.calc_df()
        print("Calculating turn frequency...")
        tfidf.calc_tf()
        tfidf.create_word2vec()
        
        tfidf.download_to_json(output_filename)
        print("Vectors were saved to " + output_filename)
        
    
    print('End!')