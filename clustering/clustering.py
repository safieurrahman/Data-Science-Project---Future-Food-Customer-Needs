import os
import sys
import math
import operator
import re
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import naive_bayes
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline

stopwords = nltk.corpus.stopwords.words("english") 

def preprocess(x):
	result = x
	result = result.lower().strip()

def preprocess_sentences(x):
	result = x
	result = re.sub('\&', 'and', result)
	result = re.sub("(?:^|\W)nan(?:$|\W)", 'None', result)
	result = re.sub('/', 'and', result)
	result = re.sub(',', ' and', result)
	result = re.sub('\.', ' and', result)
	result = re.sub('\W+', ' ', result)
	result = result.lower().strip()
	return result

def preprocess_comfort_food(x):
    result = re.sub('/', ',', x)
    result = re.sub("(?:^|\W)and(?:$|\W)", ', ', result)
    result = re.sub('nan', '', result)
    result = re.sub('\.', '', result)
    result = re.sub('\r', '', result)
    # for mac & cheese
    result = re.sub('\&', 'n', result)
    result = result.lower().strip()
    return result

tokenize = lambda x : [s.strip() for s in x.split(',') if s != '']
tokenize_space = lambda x: [s.strip() for s in x.split(' ')]

def get_features(data, column_name, tokenize=tokenize, preprocessor= preprocess):
	vectorizer = TfidfVectorizer( 
		tokenizer=tokenize, 
		stop_words=stopwords, 
		use_idf=True, 
		smooth_idf=False, 
		norm=None, 
		decode_error='replace', 
		max_features=10,
		preprocessor=preprocess, 
		lowercase=True 
	)
	tfidf = vectorizer.fit_transform(data[column_name].values.astype('U'))
	vocab = {v:i for i, v in enumerate(vectorizer.get_feature_names())}
	idf_vals = vectorizer.idf_
	idf_dict = {i:idf_vals[i] for i in vocab.values()}
	return vocab


folder_path = '.'
food_coded_path = os.path.join(folder_path, 'food_coded.csv')
food_coded_data = pd.read_csv(food_coded_path)

def clean_floats(x):
	try:
		result = float(x)
	except:
		result = np.nan
	return result

def clean_numeric_column(data, column_name):
	imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
	result = imputer.fit_transform(np.array(data[column_name].apply(clean_floats)).reshape(-1,1)).flatten()
	return pd.Series(result)

columns = ['comfort_food', 'comfort_food_reasons', 'diet_current', 'eating_changes', 
'food_childhood', 'healthy_meal', 'ideal_diet', 'meals_dinner_friend']

def get_word_frequence_comma_sep(column, ngram_range=(1,1)):
	corpus = food_coded_data[column].values.astype('U')
	cv = CountVectorizer(
		tokenizer=tokenize_space, preprocessor=preprocess_sentences, 
		ngram_range=(1, 1), stop_words=stopwords
	)
	vec = cv.fit(corpus)

	bag_of_words = vec.transform(corpus) 
	sum_words = bag_of_words.sum(axis=0)  
	words_freq = [(word, sum_words[0, idx]) for word, idx in    
	vec.vocabulary_.items()] 
	words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
	return words_freq 


def get_tfidf_weights(column, ngram_range=(1,4), max_features=100):
	corpus = food_coded_data[column].values.astype('U')
	tfv = TfidfVectorizer(ngram_range=ngram_range, preprocessor=preprocess_sentences, 
		tokenizer=tokenize_space, max_features=max_features, 
		max_df=1, stop_words=stopwords
	)
	tfv.fit_transform(corpus)
	features = dict(zip(tfv.get_feature_names(), tfv.idf_))
	return features 
