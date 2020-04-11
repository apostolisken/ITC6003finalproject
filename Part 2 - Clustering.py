# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 19:34:49 2020

@author: apost
"""
#%matplotlib qt

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.cluster import DBSCAN
from matplotlib import cm as cm
from sklearn.decomposition import PCA
from IPython.display import display # Allows the use of display() for DataFrames
from sklearn.neighbors import NearestNeighbors


data = pd.read_csv('Wholesale customers data.csv')

#check for na
missingCols=data.columns[data.isna().any()]

#########################################################################################################
# CORRELATION HEATMAP

corr = data.corr()

#fig, ax = plt.subplots(1, 1, figsize=(4,4))

plt.figure(10)
hm = sns.heatmap(corr, 
   #              ax=ax,           # Axes in which to draw the plot, otherwise use the currently-active Axes.
                 cmap="coolwarm", # Color Map.
                 square=True,    # If True, set the Axes aspect to “equal” so each cell will be square-shaped.
                 annot=True, 
                 fmt='.2f',       # String formatting code to use when adding annotations.
                 #annot_kws={"size": 14},
                 linewidths=.5)

#########################################################################################################

#drop Channel and Region columns
data_clust=data.copy().drop(['Channel','Region'],axis=1)

#########################################################################################################
# Feature Scaling

# # Scale the data using the natural logarithm
log_data_clust = np.log(data_clust)

plt.figure(11)
# Produce a scatter matrix for each pair of newly-transformed features
pd.plotting.scatter_matrix(log_data_clust, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

#########################################################################################################
# Outliers IQR method

Q1=log_data_clust.quantile(0.25)
Q3=log_data_clust.quantile(0.75)
IQR = Q3 - Q1
IQR_df=pd.DataFrame(IQR)
outliers_df=(((log_data_clust<(Q1-1.5 * IQR))|(log_data_clust>(Q3+1.5*IQR))).any(axis=1))
data_clust_IQR= log_data_clust[~((log_data_clust<(Q1-1.5 * IQR))|(log_data_clust>(Q3+1.5*IQR))).any(axis=1)]

#########################################################################################################
# PCA Dimensionality reduction
pca = PCA(n_components=6)
pca.fit(data_clust_IQR)

def pca_results(good_data, pca):
	'''
	Create a DataFrame of the PCA results
	Includes dimension feature weights and explained variance
	Visualizes the PCA results
	'''

	# Dimension indexing
	dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

	# PCA components
	components = pd.DataFrame(np.round(pca.components_, 4), columns = list(good_data.keys()))
	components.index = dimensions

	# PCA explained variance
	ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
	variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
	variance_ratios.index = dimensions

	# Create a bar plot visualization
	fig, ax = plt.subplots(figsize = (14,8))

	# Plot the feature weights as a function of the components
	components.plot(ax = ax, kind = 'bar');
	ax.set_ylabel("Feature Weights")
	ax.set_xticklabels(dimensions, rotation=0)


	# Display the explained variance ratios
	for i, ev in enumerate(pca.explained_variance_ratio_):
		ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

	# Return a concatenated DataFrame
	return pd.concat([variance_ratios, components], axis = 1)

pca_results = pca_results(data_clust_IQR, pca)
# DataFrame of results
display(pca_results)

# Cumulative explained variance should add to 1
display(pca_results['Explained Variance'].cumsum())

pca=PCA(n_components=2, random_state=0)

data_clust_reduced = pd.DataFrame(pca.fit_transform(data_clust_IQR),columns=['dimension1','dimension2'])

#########################################################################################################
#Creating Clusters with Gaussian Mixture

silhouettesAll_GM=[]

for n in range(2,7):
    # Apply clustering algorithm
    GMclusterer = mixture.GaussianMixture(n_components=n,covariance_type='full')
    GMclusterer.fit(data_clust_IQR)

    # Predict the cluster for each data point
    GMpreds = GMclusterer.predict(data_clust_IQR)
    
    # Find the cluster centers
    GMcenters = GMclusterer.fit(data_clust_reduced).means_
    #evalute
    silhouette_values = silhouette_samples(data_clust_IQR,GMpreds)
    print ('GM silhouette score for',n,'clusters=', np.mean(silhouette_values))
    #Visualize cluster
    plt.figure(n)
    plt.scatter(data_clust_reduced['dimension1'], data_clust_reduced['dimension2'], c=GMpreds, s=20, cmap='coolwarm')
    centers = GMcenters
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.title('Gaussian Mixture Clustering with 2 dimensions and '+str(n)+' clusters')
    
    silhouettesAll_GM.append(np.mean(silhouette_values))    

#Silhouete plot
plt.figure()
plt.plot(range(2,7),silhouettesAll_GM,'r*-')
plt.ylabel('GM Silhouette score')
plt.xlabel('GM Number of clusters')

#True Centers for 2 Clusters for Gaussian Mixture
# Apply clustering algorithm
GMclusterer = mixture.GaussianMixture(n_components=2,covariance_type='full')
GMclusterer.fit(data_clust_IQR)

# Predict the cluster for each data point
GMpreds = GMclusterer.predict(data_clust_IQR)
    
# Find the cluster centers
GMcenters = GMclusterer.fit(data_clust_reduced).means_

# Inverse transform the centers
GM_log_centers =pca.inverse_transform(GMcenters)

# Exponentiate the centers
GM_true_centers = np.exp(GM_log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(GMcenters))]
GM_true_centers_df = pd.DataFrame(np.round(GM_true_centers), columns = data_clust.keys())
GM_true_centers_df.index = segments
display(GM_true_centers_df)

#########################################################################################################
#Creating Clusters with K-Means

silhouettesAll_KM=[]
inertiasAll_KM=[]

for n in range(2,7):
    # Apply clustering algorithm
    KMclusterer =KMeans(n_clusters=n)
    KMclusterer.fit(data_clust_IQR)

    # Predict the cluster for each data point
    KMpreds = KMclusterer.predict(data_clust_IQR)
    
    # Find the cluster centers
    KMcenters = KMclusterer.fit(data_clust_reduced).cluster_centers_
    #evalute
    silhouette_values = silhouette_samples(data_clust_IQR,KMpreds)
    print ('KM silhouette score for',n,'clusters=', np.mean(silhouette_values))
    #Visualize cluster
    plt.figure(n)
    plt.scatter(data_clust_reduced['dimension1'], data_clust_reduced['dimension2'], c=KMpreds, s=20, cmap='coolwarm')
    centers = KMcenters
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.title('K-Means Clustering with 2 dimensions and '+str(n)+' clusters')
    
    inertiasAll_KM.append(KMclusterer.inertia_)
    silhouettesAll_KM.append(np.mean(silhouette_values))    

#Silhouete plot
plt.figure()
plt.plot(range(2,7),silhouettesAll_KM,'r*-')
plt.ylabel('KM Silhouette score')
plt.xlabel('KM Number of clusters')

#Silhouete plot
plt.figure()
plt.plot(range(2,7),inertiasAll_KM,'g*-')
plt.ylabel('KM Inertia score')
plt.xlabel('KM Number of clusters')

#True Centers for 2 Clusters for K-Means
# Apply clustering algorithm
KMclusterer = KMeans(n_clusters=2)
KMclusterer.fit(data_clust_IQR)

# Predict the cluster for each data point
KMpreds = KMclusterer.predict(data_clust_IQR)
    
# Find the cluster centers
KMcenters = KMclusterer.fit(data_clust_reduced).cluster_centers_

# Inverse transform the centers
KM_log_centers =pca.inverse_transform(KMcenters)

# Exponentiate the centers
KM_true_centers = np.exp(KM_log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(KMcenters))]
KM_true_centers_df = pd.DataFrame(np.round(KM_true_centers), columns = data_clust.keys())
KM_true_centers_df.index = segments
display(KM_true_centers_df)

#########################################################################################################
#Creating Clusters with DBSCAN


#finding number of eps
neigh = NearestNeighbors(n_neighbors=100)
nbrs = neigh.fit(data_clust_reduced)
distances, indices = nbrs.kneighbors(data_clust_reduced)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)

db = DBSCAN(eps=0.4, min_samples=7).fit(data_clust_reduced)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# print cluster labels. The value -1 means it's outside all clusters
labels = db.labels_
print (labels)
 
#evaluate with the silhouette criteian
silhouette_values = silhouette_samples(data_clust_reduced, labels)
print ('silhouette=', np.mean(silhouette_values))

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels))- (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(data_clust_reduced, labels))


plt.figure(20)

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    # core nodes
    xy = data_clust_reduced[class_member_mask & core_samples_mask]
    plt.plot(xy['dimension1'], xy['dimension2'],'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=10)

    # border nodes
    xy = data_clust_reduced[class_member_mask & ~core_samples_mask]
    plt.plot(xy['dimension1'], xy['dimension2'], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=4)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()






















