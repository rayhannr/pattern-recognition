import re
import numpy as np
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import sent_tokenize, word_tokenize

## Preprocessing
corpus = [] #array containing words in each document
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

#Seeding the bag of words, containing all words in all abstracts uniquely
bag_of_words = []
for document in corpus:
    bag_of_words = np.concatenate((bag_of_words, document), axis=None)
    bag_of_words  = np.unique(bag_of_words)
bag_of_words = bag_of_words.reshape(1, -1)

len(set(corpus[0]))
len(np.unique(corpus[0]))

## Calculating Count Vectorizer
cv = np.zeros((bag_of_words.shape[1], 10))
for i in range(len(corpus)):
    for j in range(len(cv)):
        cv[j, i] = corpus[i].count(bag_of_words[0, j])
cv = cv.T

## Calculating Jaccard Similarity
def jaccard_similarity(doc1, doc2):
    union = 0
    intersection = 0
    for i in range(len(doc1)):
        if doc1[i] > 0 and doc2[i] > 0:
            intersection += 1
            union += 1
        elif doc1[i] > 0 or doc2[i] > 0:
            union += 1
    
    return intersection / union
jaccard_similarity(cv[2], cv[3])

#Creating matrix showing similarity of each document
doc_similarity = np.zeros((len(corpus), len(corpus)))
for i in range(doc_similarity.shape[0]):
    doc_similarity[i, i] = 1
    for j in range(i+1, doc_similarity.shape[1]):
        doc_similarity[i, j] = jaccard_similarity(cv[i], cv[j])
        doc_similarity[j, i] = doc_similarity[i, j]


    

