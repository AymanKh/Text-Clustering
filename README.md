# Text Clustering

Installation:
After cloning the repo, run the following command:

python Clustering_Papers.py

make sure your are using python 3.5 and have data scinece packages installed. Below is a list of the imports needed:
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

Project Motivation:
This project is motivated by the need to create a model that finds publicatoins similar to publication a graduate student has read. The goal is to reduce the time the student needs to find other highly correlated publications. In addition, this model enables students to discover new research topics that they may have not been aware of

File descriptions:
Clustering_Papers.py -- the file containing the python code
paper_dataset.text -- the file containg the papers' data, this file will be parsed by the python script
README.md -- THIS FILE

Results:
for a sample result, open ward_clusters_new.png to examine a dendogram that shows the liklihood similarity between different papers. The abstract of every paper has been used to avail this visualization.

Acknowledgments:
The following articles have been used for inspiration:
https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/
https://medium.com/@datamonsters/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908




CRISP-DM:

- Business Understanding: The goal of this project is to develop an algorithm that enables graduate students to find research papers that are similar to paper they liked. In addition, it enables graduate students to view common categories papers fall under
- Data Understanding: Our dataset contains multiple rows, each row corresponding to one paper. The columns are:
    a) Paper Title
    b) Paper Abstract
    c) Paper conference
    d) Paper authors
    e) Paper Date
    f) Paper ID
- Data Preperation: We only use a) and b) for this project, since the other columns are not needed as per my assesment
- Modeling: We use text preprocessing techniques, which are tokenizing, stemming, removing stop words. After that, we build an IDF-DF matrix and then we build a model using a clustering algorithm to connect a paper to its neighbours
- Evaluation: since this is an unsupervised learning problem, We evaluate our model by examining the cluster and jumping to data-driven conclusions regarding the property of every cluster
- Deployment: This code can be run by any python compile as long as the python installation contains the needed packages

Note on Missing values:
- All rows containing missing values have been dropped, this is due to:
    - There are so few of them
    - since this is text processing, replacing missing values is not numerically possible

Note on Categorical variables:
- Since this is a text-processing project, There wasn't a need to deal with categorical variables

