import re
import numpy as np
import matplotlib.pyplot as plt
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize 

corpus = [] #array containing words and the TF inside each abstract
bag_of_words = []
for i in range(2):
    abstract_file = open("abstrak{}.txt".format(i+1), "r")
    abstract_words = []
    stopwords = StopWordRemoverFactory().create_stop_word_remover()
    stemmer = StemmerFactory().create_stemmer()
    
    #extracting words in every line of the selected abstract
    for line in abstract_file:
        if line.strip():
            sentence = sent_tokenize(line) #making array of sentences
            for word in sentence:
                word = word.replace('\n', '').lower() #hapus newline dan buat lowercase
                word = re.sub(r'[^a-zA-Z]',' ', word) #hapus angka
                word = stopwords.remove(word) #remove stopwords
                word = stemmer.stem(word) #stem kata ke mashdarnya
                word = word_tokenize(word) #split kalimat jadi tiap array of words
                abstract_words.append(word)           
    abstract_file.close()
    
    #combining line of words in abstract into one, then adding it into corpus
    combined_line = []
    for line in abstract_words:
        combined_line += line
    corpus.append(combined_line)
    
#Seeding the bag of words, containing all words in all abstracts uniquely
for abstract in corpus:
    bag_of_words += abstract
    bag_of_words = np.unique(bag_of_words)
    
def term_frequency(abstract_list, element):
    return abstract_list.count(element)

for i in range(len(corpus)):
    tf = []
    for word in bag_of_words:
        tf.append(term_frequency(corpus[i], word))
    corpus[i] = np.stack((bag_of_words, tf)).T