"""
Clustering Visualization Module.

This module provides functions to visualize clustering results using 3D plots.

Functions:
    plot_clusters_3d: Creates a 3D scatter plot of clustered data.
    plot_kmeans: Plots KMeans clustering results.
    plot_gmm: Plots Gaussian Mixture Model clustering results.
    plot_dbscan: Plots DBSCAN clustering results.
    plot_ensemble: Plots Ensemble clustering results.
    plot_agglomerative: Plots Agglomerative clustering results.
    plot_optics: Plots OPTICS clustering results.
    plot_hdbscan: Plots HDBSCAN clustering results.
"""


from typing import Dict
from clearml import Task, Logger
from src.config import (
    np, plt, Axes3D, PCA,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)

# def plot_clusters_3d(features_scaled: np.ndarray, labels: np.ndarray, title: str, task: Task):
#     pca = PCA(n_components=3)
#     pca_result = pca.fit_transform(features_scaled)

#     # Convert labels to a list of strings
#     label_list = [str(label) for label in labels]

#     # Create the scatter plot data
#     scatter_data = pca_result.tolist()  # Convert to list of lists

#     # Ensure all elements in scatter_data are numerical
#     scatter_data = [[float(val) for val in point] for point in scatter_data]

#     # Report the 3D scatter plot using ClearML
#     Logger.current_logger().report_scatter3d(
#         title=title,
#         series="cluster_scatter",
#         iteration=0,
#         scatter=scatter_data,
#         labels=label_list,
#         xaxis="PCA Component 1",
#         yaxis="PCA Component 2",
#         zaxis="PCA Component 3"
#     )
from typing import Dict
from clearml import Task, Logger
from src.config import (
    np, plt, Axes3D, PCA,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)

# def plot_clusters_3d(features_scaled: np.ndarray, labels: np.ndarray, title: str, task: Task):
#     pca = PCA(n_components=3)
#     pca_result = pca.fit_transform(features_scaled)
    
#     # Convert labels to a list of strings
#     label_list = [str(label) for label in labels]
    
#     # Create the scatter plot data
#     scatter_data = pca_result.tolist()  # Convert to list of lists
    
#     # Report the 3D scatter plot using ClearML
#     Logger.current_logger().report_scatter3d(
#         title=title,
#         series="cluster_scatter",
#         iteration=0,
#         scatter=scatter_data,
#         labels=label_list,
#         xaxis="PCA Component 1",
#         yaxis="PCA Component 2",
#         zaxis="PCA Component 3"
#     )
# def plot_clusters_3d(features_scaled: np.ndarray, labels: np.ndarray, title: str, logger):
#     """Generates a 3D scatter plot of clustered data using PCA for dimensionality reduction."""
#     pca = PCA(n_components=3)
#     pca_result = pca.fit_transform(features_scaled)
    
#     # Create figure for plotting
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2],
#                          c=labels, cmap='viridis', alpha=0.6, edgecolors='w', s=50)
    
#     ax.set_title(title)
#     ax.set_xlabel('PCA Component 1')
#     ax.set_ylabel('PCA Component 2')
#     ax.set_zlabel('PCA Component 3')
#     plt.colorbar(scatter, label='Cluster label')
#     plt.close(fig)  # Close the figure to prevent it from displaying in notebooks or environments

#     # Convert the PCA results and labels to list for ClearML logging
#     scatter_data = pca_result.tolist()
#     label_list = [str(label) for label in labels]  # Ensure labels are string for ClearML

#     # Log 3D scatter plot to ClearML
#     logger.report_scatter3d(
#         title=title,
#         series='3D PCA Clusters',
#         iteration=0,
#         scatter=scatter_data,
#         labels=label_list,
#         xaxis='PCA Component 1',
#         yaxis='PCA Component 2',
#         zaxis='PCA Component 3'
#     )

from clearml import Task, Logger
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def plot_clusters_3d(features_scaled: np.ndarray, labels: np.ndarray, title: str, task: Task):
    # Initialize PCA and transform features
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_scaled)
    
    # Fetch the logger from the task
    logger = task.get_logger()

    # Create figure for the scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2],
                         c=labels, cmap='viridis', alpha=0.6, edgecolors='w', s=50)
    
    ax.set_title(title)
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    plt.colorbar(scatter, label='Cluster label')

    # Close the plot to save resources
    plt.close()

    # Prepare data for ClearML logging
    scatter_data = pca_result.tolist()
    label_list = [str(label) for label in labels]

    # Log the 3D scatter plot
    logger.report_scatter3d(
        title=title,
        series='3D PCA Clusters',
        iteration=0,
        scatter=scatter_data,
        labels=label_list,
        xaxis='PCA Component 1',
        yaxis='PCA Component 2',
        zaxis='PCA Component 3'
    )

def plot_kmeans(features_scaled: np.ndarray, labels: np.ndarray, task: Task):
    """Plot KMeans clustering results."""
    plot_clusters_3d(features_scaled, labels, 'KMeans Clustering with 3D PCA', task)

def plot_gmm(features_scaled: np.ndarray, labels: np.ndarray, task: Task):
    """Plot Gaussian Mixture Model clustering results."""
    plot_clusters_3d(features_scaled, labels, 'Gaussian Mixture Model Clustering with 3D PCA', task)

def plot_dbscan(features_scaled: np.ndarray, labels: np.ndarray, task: Task):
    """Plot DBSCAN clustering results."""
    plot_clusters_3d(features_scaled, labels, 'DBSCAN Clustering with 3D PCA', task)

def plot_ensemble(features_scaled: np.ndarray, labels: np.ndarray, task: Task):
    """Plot Ensemble clustering results."""
    plot_clusters_3d(features_scaled, labels, 'Ensemble Clustering with 3D PCA', task)

def plot_agglomerative(features_scaled: np.ndarray, labels: np.ndarray, task: Task):
    """Plot Agglomerative clustering results."""
    plot_clusters_3d(features_scaled, labels, 'Agglomerative Clustering with 3D PCA', task)

def plot_optics(features_scaled: np.ndarray, labels: np.ndarray, task: Task):
    """Plot OPTICS clustering results."""
    plot_clusters_3d(features_scaled, labels, 'OPTICS Clustering with 3D PCA', task)

def plot_hdbscan(features_scaled: np.ndarray, labels: np.ndarray, task: Task):
    """Plot HDBSCAN clustering results."""
    plot_clusters_3d(features_scaled, labels, 'HDBSCAN Clustering with 3D PCA', task)
  
    
def plot_kmeans(features_scaled: np.ndarray, labels: np.ndarray, logger):
    plot_clusters_3d(features_scaled, labels, 'KMeans Clustering with 3D PCA', logger)

def plot_gmm(features_scaled: np.ndarray, labels: np.ndarray, logger):
    plot_clusters_3d(features_scaled, labels, 'Gaussian Mixture Model Clustering with 3D PCA', logger)

def plot_dbscan(features_scaled: np.ndarray, labels: np.ndarray, logger):
    plot_clusters_3d(features_scaled, labels, 'DBSCAN Clustering with 3D PCA', logger)

def plot_ensemble(features_scaled: np.ndarray, labels: np.ndarray, logger):
    plot_clusters_3d(features_scaled, labels, 'Ensemble Clustering with 3D PCA', logger)

def plot_agglomerative(features_scaled: np.ndarray, labels: np.ndarray, logger):
    plot_clusters_3d(features_scaled, labels, 'Agglomerative Clustering with 3D PCA', logger)

def plot_optics(features_scaled: np.ndarray, labels: np.ndarray, logger):
    plot_clusters_3d(features_scaled, labels, 'OPTICS Clustering with 3D PCA', logger)

def plot_hdbscan(features_scaled: np.ndarray, labels: np.ndarray, logger):
    plot_clusters_3d(features_scaled, labels, 'HDBSCAN Clustering with 3D PCA', logger)
    
    




