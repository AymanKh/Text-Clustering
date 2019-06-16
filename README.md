# Text Clustering

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