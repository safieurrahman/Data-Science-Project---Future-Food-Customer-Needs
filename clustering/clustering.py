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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

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

def clean_numeric(x, function='int'):
	try:
		if function == 'float':
			result = float(x)
		else: 
			result = int(x)
	except:
		result = np.nan
	return result

def clean_numeric_column(data, column_name, function='int'):
	imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
	result = imputer.fit_transform(np.array(data[column_name].apply(lambda x: clean_numeric(x, function))).reshape(-1,1)).flatten()
	return pd.Series(result)

def clean_all_columns(data):
	result = data.copy()
	for x in [('GPA', 'float'), 'Gender', 'breakfast', 'calories_chicken',	'calories_day', 'drink', 'calories_scone', 'coffee', 
	'comfort_food_reasons_coded', 'cook', 'comfort_food_reasons_coded', 'cuisine', 'diet_current_coded', 'eating_changes_coded',
	'eating_changes_coded1', 'eating_out', 'employment', 'ethnic_food', 'exercise', 'father_education', 'fav_cuisine_coded', 'fav_food',
	'fries', 'fruit_day', 'grade_level', 'greek_food', 'ideal_diet_coded', 'income', 'indian_food', 'italian_food', 'life_rewarding', 'marital_status', 
	'mother_education', 'nutritional_check', 'on_off_campus', 'parents_cook', 'pay_meal_out', 'persian_food', 'self_perception_weight', 'soup', 'sports', 'thai_food', 'tortilla_calories', 'turkey_calories',
	'veggies_day', 'vitamins', 'waffle_calories', 'weight']:
		if type(x) == tuple:
			clean_column = clean_numeric_column(result, x[0], x[1])
			result.drop(labels=x[0], axis='columns', inplace=True)
			result[x[0]] = clean_column
		else:
			clean_column = clean_numeric_column(result, x, 'int')
			result.drop(labels=x, axis='columns', inplace=True)
			result[x] = clean_column
	return result

columns = ['comfort_food', 'comfort_food_reasons', 'diet_current', 'eating_changes', 
'food_childhood', 'healthy_meal', 'ideal_diet', 'meals_dinner_friend']

def get_word_frequence_comma_sep(column, ngram_range=(1,1)):
	corpus = food_coded_data[column].values.astype('U')
	cv = CountVectorizer(
		tokenizer=tokenize, preprocessor=preprocess_sentences, 
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

# This shows that its difficult to cluster the comfort food via simple tfidf
def get_comfort_food_cluster_words(n_clusters):
	corpus = food_coded_data['comfort_food'].values.astype('U')
	tfv = TfidfVectorizer(ngram_range=(1,1), preprocessor=preprocess_comfort_food, 
		tokenizer=tokenize
	)
	tfidf = tfv.fit_transform(corpus)
	kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1).fit(tfidf)
	order_centroids = kmeans.cluster_centers_.copy().argsort()[:, ::-1]
	terms = tfv.get_feature_names()
	for i in range(n_clusters):
		print("Cluster %d" % i)
		for ind in order_centroids[i, :10]:
			print("\t%s"% terms[ind])

def get_comfort_food_cluster_elbow(max_clusters):
	corpus = food_coded_data['comfort_food'].values.astype('U')
	tfv = TfidfVectorizer(ngram_range=(1,1), preprocessor=preprocess_comfort_food, 
		tokenizer=tokenize
	)
	tfidf = tfv.fit_transform(corpus)
	cluster_range = range(1, max_clusters)
	cluster_errors = []
	for num_clusters in cluster_range:
		clusters = KMeans(num_clusters,init='k-means++', max_iter=100, n_init=1)
		clusters.fit(tfidf)
		cluster_errors.append( clusters.inertia_ )
	clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

	plt.figure(figsize=(12,6))
	plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
	plt.show()

def get_comfort_food_clusters(n_clusters):
	corpus = food_coded_data['comfort_food'].values.astype('U')
	tfv = TfidfVectorizer(ngram_range=(1,1), preprocessor=preprocess_comfort_food, 
		tokenizer=tokenize
	)
	tfidf = tfv.fit_transform(corpus)
	return KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1).fit_predict(tfidf)
	


def get_full_clusters(data, max_clusters, selected_columns):
	clean_data = clean_all_columns(data)
	clean_data['comfort_food_coded'] = get_comfort_food_clusters(20)
	n_clean_data = clean_data[selected_columns]
	mms = MinMaxScaler()
	mms.fit(n_clean_data)
	transformed_data = mms.transform(n_clean_data)
	Sum_of_squared_distances = []
	K = range(1,max_clusters)
	for k in K:
	    km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
	    km = km.fit(transformed_data)
	    Sum_of_squared_distances.append(km.inertia_)
	plt.plot(K, Sum_of_squared_distances, 'bx-')
	plt.xlabel('k')
	plt.ylabel('Sum_of_squared_distances')
	plt.title('Elbow Method For Optimal k')
	plt.show()
	# we see that 13 or 14 are good clusters
	true_k = 14
	labels = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1).fit_predict(transformed_data)
	return pd.DataFrame(np.c_[transformed_data, labels], columns=selected_columns+['label'])


def get_clusters_results_summary(data):
		for i in range(0, 14):
			csummary = data[data.label == i].mean().drop('label')
			res = 'For Cluster %s:\n' % i 
			res += csummary.__str__()
			res = re.sub('dtype: float64', '', res)
			res +='\n'
			print(res)
			csummary.plot.bar()
			plt.show()

def main():
	selected_columns = ['Gender',
		'calories_day',
		'comfort_food_coded',
		'comfort_food_reasons_coded',
		'cook',
		'diet_current_coded',
		'eating_changes_coded',
		'eating_changes_coded1',
		'eating_out',
		'ethnic_food',
		'exercise',
		'fav_cuisine_coded',
		'fruit_day',
		'healthy_feeling',
		'ideal_diet_coded',
		'nutritional_check',
		'sports',
		'vitamins'
	]
	clusters = get_full_clusters(food_coded_data, 14, selected_columns)
	get_clusters_results_summary(clusters)

if __name__ == '__main__':
	main()


