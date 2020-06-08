# Importing the required libraries 
import numpy as np 
import pandas as pd from sklearn.feature_extraction.text 
import TfidfVectorizer from sklearn.metrics.pairwise 
import sigmoid_kernel import sys # load the dataset 


js = pd.read_csv("all_articles_blogs.csv", encoding='latin-1') # function to preprocess the dataset 
def preprocessing(): 
    # Model defination with the feature declaration 
    tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 5), stop_words='english') 
    # Filling NaNs with empty string 
    js['ArticleTitle'] = js['ArticleTitle'].fillna('') 
    tfv_matrix = tfv.fit_transform(js['ArticleTitle']) 
    # Compute the sigmoid kernel
    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
    # Generate the indices for the recommender system, removing the duplicates
    indices = pd.Series(js.index, index=js['ArticleFullPath'
    ]).drop_duplicates()
    #returning the indices and sigmoid kernel matrix
    return indices, sig
    
def give_rec(title):
    indices, sig = preprocessing()
    # Get the index corresponding to original_title
    idx = indices[title]
    # Get the pairwsie similarity scores
    sig_scores = list(enumerate(sig[idx]))
    # Sort the Articles
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    # Scores of the 5 most similar articles
    sig_scores = sig_scores[1:6]
    # Movie indices
    movie_indices = [i[0] for i in sig_scores]
    # Top 10 most similar movies
    return js['ArticleFullPath'].iloc[movie_indices]
    
# recommender control logic
def recommender(link):
    result = give_rec(link)
    for i in result.index:
    print(js['ArticleFullPath'][i])

if __name__ == "__main__":
    recommender(argv[1])