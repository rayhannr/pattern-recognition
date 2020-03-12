import re
import numpy as np
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize 

## Preprocessing
corpus = [] #array containing words and the CV inside each abstract
for i in range(10):
    abstract_file = open("abstrak{}.txt".format(i+1), "r", encoding="utf-8")
    abstract_words = []
    factory = StopWordRemoverFactory()
    stopwords = StopWordRemoverFactory().create_stop_word_remover()
    stemmer = StemmerFactory().create_stemmer()
    
    #extracting words in every line of the selected abstract
    for line in abstract_file:
        if line.strip():
            sentence = sent_tokenize(line)
            for word in sentence:
                word = word.lower()
                word = re.sub(r'[^a-zA-Z]',' ', word)
                word = stemmer.stem(word)
                word = word_tokenize(word)
                word = [w for w in word if not w in factory.get_stop_words()]
                abstract_words += word          
    abstract_file.close()
    
    corpus.append(abstract_words)

## Calculating the Term Frequency    
def term_frequency(document, word):
    return document.count(word)
    
#Seeding the bag of words, containing all words in all abstracts uniquely
bag_of_words = []
for document in corpus:
    bag_of_words = np.concatenate((bag_of_words, document), axis=None)
    bag_of_words  = np.unique(bag_of_words)

#Seeding the bag of words with its TF for each document
tf = np.zeros((bag_of_words.shape[0], 10))
for i in range(len(corpus)):
    for j in range(len(tf)):
        tf[j, i] = term_frequency(corpus[i], bag_of_words[j])
        
    #Normalizing the TF
    tf[:, i] /= np.sum(tf[:, i])
    
    
## Calculating the Inverse Document Frequency
def document_frequency(document, word, count):
    if word in document:
        count += 1
    return count

def inverse_document_frequency(df):
    document_length = 10
    return np.log(document_length/(df + 1))

idf = np.zeros((tf.shape[0], 1))
for i in range(len(bag_of_words)):
    for document in corpus:
        idf[i, 0] = document_frequency(document, bag_of_words[i], idf[i, 0])
    idf[i, 0] = inverse_document_frequency(idf[i, 0])

#Calculating Wdt = tf * idf
wdt = tf.copy()
wdt = np.multiply(wdt, idf)