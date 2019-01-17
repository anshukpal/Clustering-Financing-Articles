# Clustering-Financing-Articles
A financial institution news agency has collected 3000 news articles that relates to several matters of financial importance. Before analyzing these unlabeled news, it is only fair to try to partition them into some sort of logical groupings based on their similarities.  The objectie of this code is to use appropriate unsupervised machine learning algorithm to form the news clusters based on their similarity. Prior to clustering, performing basic natural language processing steps such as stemming, tokenization and word vectorization for best results.

# Data Analysis and Approach

1. Understand the data content 
2. Performing Basic Stemming on the text content using PorterStemmer
3. Performing Tokeinization and Vectorization - using CountVectorizer and TfidfTransformer
4. Scaling and Dimension Reduction - using StandardScaler and TruncatedSVD
5. Elbow Analysis - Understanding the range of clusters and plotting the same using matplotlib.pyplot
6. Based on Elbow Analysis ,finding average silhouette_score for every cluster formation 
7. Performing KMeans clustering and saving the clusters into a csa=v file
