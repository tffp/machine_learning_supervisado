#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 18:01:56 2022

@author: Teresa
"""

import pandas as pd
import numpy as np
from numpy import char
import matplotlib.pyplot as plt
import statistics as stats
import sklearn
from  sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer 
import math
import fnmatch
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = stopwords.words('spanish')


"""
Funciones previas:
    
        Elaboramos funciones correspondientes al preprocessing del texto dado
        Y para la lectura de los ficheros .txt
"""
def doc_reader(file_name):
    with open(file_name) as f:
        text = f.read()
    return text


def convert_lower_case(data):
    data = np.array(data)
    data_lower = np.char.lower(data)
    return data_lower

#stopwords
def remove_stop_words(data):
    new_text = ""
    for word in data: 
        if word not in stop_words:
            new_text = new_text +" "+ word
    return new_text.split(' ')

#símbolos  
def remove_punctuation(data):
    symbols= "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in symbols:
        data = np.char.replace(data,i,' ')
    return data
    
# palabras de una letra - poco informativas
def remove_single_characters(data):
    #new_text= ""
    for w in data:
        if len(w) <= 1:
            data = np.char.replace(data,w,' ')
            #new_text = new_text + " " + w
    return data
    
def stemming(data):
    new_text= ""
    stemmer= SnowballStemmer('spanish')
    for w in data:
        new_text = new_text + ' ' + str(stemmer.stem(w))
    return new_text.split(' ')


# preprocessing completo
def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data)
    data = remove_single_characters(data)
    data = remove_stop_words(data)
    data = stemming(data)
    new_text = ""
    for w in data:
            new_text = new_text + ' ' + w
    new_text = new_text.strip()
    data = new_text.split(' ')
    return data




"""
Funcion 1:
    
        Elaboramos la parte de la función que hace la parte
        tf del indice : nº de veces que aparece la palabra/nº  de 
        palabras del documento
        
"""
def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word,count in wordDict.items():
        tfDict[word] =  count / float(bagOfWordsCount)
    return tfDict

"""
Funcion 2:
    Elaboramos la parte de la función que hace la parte idf del índice
    El logaritmo del nº de documentos dividido entre el nº de documentos
    que contienen una palabra concreta 'w'.
    
"""


def computeIDF(numOfWordsTextos):
    N  = len(numOfWordsTextos)
    idfDict = dict.fromkeys(numOfWordsTextos[0].keys(),0)
    for texto in numOfWordsTextos:
        for word, val in texto.items():
            if val > 0:
                idfDict[word] += 1
    
    for word,val in idfDict.items():
        idfDict[word] = math.log(N/float(val))
    return idfDict


"""
Funcion 3:
    Finalmente obtenemos el indice TF-IDF multiplicando ambos
    
"""

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word,val in tfBagOfWords.items():
        tfidf[word] = val*idfs[word]
    return tfidf


