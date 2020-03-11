import re
import string
import matplotlib.pyplot as plt
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize 

corpus = [] #array containing words inside each abstract
for i in range(2):
    input_file = open("abstrak{}.txt".format(i+1), "r")
    abstract = []
    stopwords = StopWordRemoverFactory().create_stop_word_remover()
    stemmer = StemmerFactory().create_stemmer()
    
    #extracting words in every line of the selected abstract
    for text in input_file:
        if text.strip():
            tokens = sent_tokenize(text) #making array of sentences
            for token in tokens:
                token = token.replace('\n', '').lower() #hapus newline dan buat lowercase
                token = re.sub(r'\d+','', token) #hapus angka
                token = token.translate(token.maketrans('','',string.punctuation)) #hapus tanda baca
                token = stopwords.remove(token) #remove stopwords
                token = stemmer.stem(token) #stem kata ke mashdarnya
                token = word_tokenize(token) #split kalimat jadi tiap array of words
                abstract.append(token)           
    input_file.close()
    
    #combining line of words in abstract into one, then adding it into corpus
    combined_line = []
    for line in abstract:
        combined_line += line
    corpus.append(combined_line)