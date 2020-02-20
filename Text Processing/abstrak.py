# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:46:56 2020

@author: Rayhan
"""
import nltk
import re
import string
import matplotlib.pyplot as plt
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist

input = open("abstrak.txt", "r")
text_line = []
stopwords = StopWordRemoverFactory().create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()
for text in input:
    if text.strip():
        tokens = nltk.tokenize.sent_tokenize(text) #making array of sentences
        for token in tokens:
            token = token.replace('\n', '').lower() #hapus newline dan buat lowercase
            token = re.sub(r'\d+','', token) #hapus angka
            token = token.translate(token.maketrans('','',string.punctuation)) #hapus tanda baca
            token = stopwords.remove(token) #remove stopwords
            token = stemmer.stem(token) #stem kata ke mashdarnya
            token = nltk.tokenize.word_tokenize(token) #split kalimat jadi tiap array of words
            text_line.append(token)

input.close()

combined_text_line = []
for line in text_line:
    combined_text_line += line
word_appearance = nltk.FreqDist(combined_text_line)
print(word_appearance.most_common())

word_appearance.plot(20,cumulative=False)
plt.show()