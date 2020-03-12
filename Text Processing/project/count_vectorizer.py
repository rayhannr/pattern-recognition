import re
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords') 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = [] #array containing words and the TF inside each abstract
for i in range(10):
    abstract_file = open("abstrak{}.txt".format(i+1), "r", encoding="utf-8")
    abstract_words = []
    ps = PorterStemmer()
    
    #extracting words in every line of the selected abstract
    for line in abstract_file:
        if line.strip():
            sentence = sent_tokenize(line)
            for words in sentence:
                words = words.lower()
                words = re.sub(r'[^a-zA-Z]',' ', words)
                words = word_tokenize(words)
                words = [ps.stem(word) for word in words if not word in set(stopwords.words('indonesian'))]
                abstract_words.append(words)           
    abstract_file.close()
    
    #combining line of words in abstract into one, then adding it into corpus
    combined_line = []
    for line in abstract_words:
        combined_line += line
    corpus.append(combined_line)
    
#Seeding the bag of words, containing all words in all abstracts uniquely
bag_of_words = []
for abstract in corpus:
    bag_of_words = np.concatenate((bag_of_words, abstract), axis=None)
    bag_of_words = np.unique(bag_of_words)
    

#Calculating TF, inserting it in corpus so the corpus will be array of words in bag of words and their TF
for i in range(len(corpus)):
    tf = []
    for word in bag_of_words:
        tf.append(corpus[i].count(word))
    corpus[i] = np.stack((bag_of_words, tf)).T