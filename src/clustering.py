"""Clustering analysis module"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def optimal_clusters(rfm, max_k=5):
    """Find optimal number of clusters using elbow method"""
    inertias = []
    silhouette_scores = []
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(rfm_scaled)
        inertias.append(kmeans.inertia_)
    
    return inertias

def cluster_customers(rfm, n_clusters=3):
    """Perform K-Means clustering"""
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    return rfm, kmeans
