# Customer Segmentation using K-Means

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def preprocess(df: pd.DataFrame, features: list) -> tuple:
    X = df[features].copy()
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler

def find_optimal_k(X: np.ndarray, max_k=10):
    wcss = []
    sil = []
    for k in range(2, max_k+1):
        km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        labels = km.fit_predict(X)
        wcss.append(km.inertia_)
        sil.append(silhouette_score(X, labels))
    return wcss, sil

def plot_elbow_silhouette(wcss, sil):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(range(2, len(wcss)+2), wcss, 'bx-', label='WCSS')
    ax2.plot(range(2, len(sil)+2), sil, 'ro-', label='Silhouette')
    ax1.set_xlabel('Number of clusters k')
    ax1.set_ylabel('WCSS')
    ax2.set_ylabel('Silhouette Score')
    plt.title('Elbow & Silhouette Analysis')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()

def train_kmeans(X: np.ndarray, k: int):
    model = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    labels = model.fit_predict(X)
    return model, labels

def visualize_clusters(df: pd.DataFrame, features: list, labels: np.ndarray, centroids: np.ndarray):
    df['Cluster'] = labels
    sns.scatterplot(x=features[0], y=features[1], hue='Cluster', palette='tab10', data=df, s=60)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')
    plt.title('Customer Segmentation')
    plt.legend()
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.show()

def summarize_clusters(df: pd.DataFrame, features: list):
    summary = df.groupby('Cluster')[features].mean().round(2)
    counts = df['Cluster'].value_counts().sort_index()
    return pd.concat([counts.rename('Count'), summary], axis=1)

def main():
    df = load_data(r"C:\Users\varsh\OneDrive\文档\customer segmentation\dataset\Mall_Customers.csv")
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    
    X, scaler = preprocess(df, features)
  
    wcss, sil = find_optimal_k(X, max_k=10)
    plot_elbow_silhouette(wcss, sil)
    
    optimal_k = 5
    model, labels = train_kmeans(X, optimal_k)
    
    visualize_clusters(df, features, labels, model.cluster_centers_)
    
    summary = summarize_clusters(df, features)
    print("Cluster Summary:\n", summary)
    
    df.to_csv('Mall_Customers_segmented.csv', index=False)

if __name__ == '__main__':
    main()

