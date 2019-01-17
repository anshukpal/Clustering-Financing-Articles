import nltk
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from collections import Counter
import time
import pandas as pd
import numpy as np
import re
from collections import Counter
from nltk.corpus import words
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import scipy.sparse

news = pd.read_csv('news.csv',encoding='utf-8-sig')

#### Stemming Component ######

stemmer = PorterStemmer()
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
progress = 0 #for keeping track of where the function is
def stem(x):
	end = time.time()
	dirty = word_tokenize(x)
	tokens = []
	for word in dirty:
		if word.strip('.') == '': #this deals with the bug
		   pass
		elif re.search(r'\d{1,}', word): #getting rid of digits
		   pass
		else:
		   tokens.append(word.strip('.'))
	global start
	global progress
	tokens = pos_tag(tokens) #
	progress += 1
	stems = ' '.join(stemmer.stem(key.lower()) for key, value in  tokens if value != 'NNP') #getting rid of proper nouns
	end = time.time()
	# print('\r {} percent, {} position, {} per second '.format(str(float(progress / len(news))), 
	# str(progress), (1 / (end - start)))) #lets us see how much time is left 
	start = time.time()
	return stems

start = time.time()
news['stems'] = news['text'].apply(lambda x: stem(x))

###### Tokeinization and Vectorization####
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(news['stems'])
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(X)

#### Standard Scaling ########33
scaler = StandardScaler(with_mean=False).fit(tfidf)
vec_matrix = scaler.fit_transform(tfidf)

#### Dimension Reduction####
pca = TruncatedSVD(n_components=100)
vec_matrix_pca = pca.fit_transform(vec_matrix)

print(vec_matrix.todense())
# print(vec_matrix)
# matrix = pd.DataFrame(vec_matrix.toarray())
np.savetxt('matr.txt', vec_matrix.todense())

# np.savetxt('vec_matrix.txt', vec_matrix)

#### Elbow Analysis #####

# cluster_range = range( 1, 20 )
# cluster_errors = []

# for num_clusters in cluster_range:
#   clusters = KMeans( num_clusters )
#   clusters.fit( X_scaled )
#   cluster_errors.append( clusters.inertia_ )

# clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

# print(clusters_df[0:10])

# plt.figure(figsize=(12,6))
# plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )

# plt.show()

##### Based on Elbow Analysis ,finding average silhouette_score for every cluster formation ####
cluster_range = range( 2, 25) ## This range was taken based elbow analysis above

for n_cluster in cluster_range:
    kmeans = KMeans(n_clusters=n_cluster).fit(vec_matrix_pca)
    label = kmeans.labels_
    sil_coeff = silhouette_score(vec_matrix_pca, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))


num_clusters = 7 #Understood from silhouette_score above
km = KMeans(n_clusters=num_clusters)
km.fit(vec_matrix_pca)
clusters = km.labels_.tolist()

news['cluster'] = clusters


print("\n")
print(news['cluster'].value_counts()) #Print the counts of doc belonging to each cluster.

del news['headline']
del news['text']
del news['stems']

print(news.shape)
news.to_csv('clusters_ver2.csv',index=False)
