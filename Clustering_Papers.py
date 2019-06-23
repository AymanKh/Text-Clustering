# from fim import apriori, eclat, fpgrowth, fim
from __future__ import print_function
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import string

import os  # for os.path.basename

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS
from scipy.cluster.hierarchy import ward, dendrogram
from spellchecker import SpellChecker

# define data path here
data_path = './paper_dataset.txt'


# This function filters a 'list of list', so that it only contains 'lists' of length 'size'
def by_size(words, size):
    """Form a list

    Keyword arguments:
    words -- a list of list containing words
    size -- The target length
    """
    return [word for word in words if len(word) == size]

# This function extract the title from data file
def parseTitle():
    """This Function opens the dataset file, and extracts the 'title' field

    Keyword arguments:
    NONE
    """
    # my code here
    with open(data_path) as f:
        lines = f.readlines()
        titles = []
        for line in lines[1:]:
            try:
                temp_titles = line.split('\t')[4]
                titles.append(temp_titles)
            except IndexError:
                continue
        # print(authors)
        return titles

# This function gets extract the abstract from data file
def parseAbstract():
    """This Function opens the dataset file, and extracts the 'Abstract' field

    Keyword arguments:
    NONE
    """
    # my code here
    with open(data_path) as f:
        lines = f.readlines()
        abstracts = []
        for line in lines[1:]:
            try:
                temp_abstracts = line.split('\t')[7][:-3]
                abstracts.append(temp_abstracts)
            except IndexError:
                continue
        return abstracts


def tokenize_and_stem(text):
    """This Function recieves a string, then tokenize and stem all words in the string

    Keyword arguments:
    text -- the string containing the target text
    """
    stemmer = SnowballStemmer("english")
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(
        text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]

    return stems


def tokenize_only(text):
    """This Function opens the dataset file, and ONLY tokenize the words

    Keyword arguments:
    text -- contains the target text
    """
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text)
              for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


if __name__ == "__main__":
    titles = parseTitle()
    abstracts = parseAbstract()

    stopwords = nltk.corpus.stopwords.words('english')
    stemmer = SnowballStemmer("english")

    totalvocab_stemmed = []
    totalvocab_tokenized = []

    # Process text to be ready for clustering. Processing include: removing punctutaion, stopwords, stemming, tokenizing
    for i in abstracts:

        replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        i = i.translate(replace_punctuation)
        allwords_stemmed = tokenize_and_stem(i)
        totalvocab_stemmed.extend(allwords_stemmed)
        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)

    
    vocab_frame = pd.DataFrame(
        {'words': totalvocab_tokenized}, index=totalvocab_stemmed)
    print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

    tfidf_vectorizer = TfidfVectorizer(max_df=0.70, max_features=200000,
                                       min_df=0.30, stop_words='english',
                                       use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(
        abstracts)  # fit the vectorizer to synopses

    print(tfidf_matrix.shape)
    terms = tfidf_vectorizer.get_feature_names()
    print("# of terms:" + str(len(terms)))
    print(terms)

    dist = 1 - cosine_similarity(tfidf_matrix)

    num_clusters = 5
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()

    papers = {'title': titles, 'abstract': abstracts, 'cluster': clusters}
    frame = pd.DataFrame(papers, index=[clusters], columns=[
                         'title', 'abstract', 'cluster'])

    print(frame['cluster'].value_counts())
    print()
    print('======================')
    print("Top terms per cluster:")
    print()

    # sort cluster centers by proximity to centroid
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    for i in range(num_clusters):
        print("Cluster %d words:" % i, end='')

        for ind in order_centroids[i, :6]:  # replace 6 with n words per cluster
            print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[
                  0][0].encode('utf-8', 'ignore'), end=',')
        print()  # add whitespace
        print()  # add whitespace


    #define the linkage_matrix using ward clustering pre-computed distances
    linkage_matrix = ward(dist) 

    # Plot the dendogram showing hierirical similarity 
    fig, ax = plt.subplots(figsize=(15, 20)) # set size
    ax = dendrogram(linkage_matrix, orientation="right", labels=titles, leaf_font_size=1)

    plt.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')

    plt.tight_layout() #show plot with tight layout

    #uncomment below to save figure
    plt.savefig('ward_clusters_new.png', dpi=800) #save figure as ward_clusters