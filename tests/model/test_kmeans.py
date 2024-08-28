"""
Test module for KMeans Clustering.

This module contains unit tests for the KMeansClusterer class
from the src.model.kmeans module.

Imports:
    - sys, os: For modifying Python path
    - numpy (as np): For numerical operations
    - pytest: For test fixtures and assertions
    - BATCH_SIZE from src.config: For setting batch size in data loading
    - get_dataloader from src.data.radar_synthetic: For creating test data
    - KMeansClusterer from src.model.kmeans: The class being tested
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config import (
    np, pytest, BATCH_SIZE
)
from src.data.radar_synthetic import get_dataloader
from src.model.kmeans import KMeansClusterer
from unittest.mock import MagicMock

@pytest.fixture
def mock_task():
    mock = MagicMock()
    mock.logger.report_scalar = MagicMock()
    mock.connect = MagicMock()
    return mock

@pytest.fixture
def dataloader():
    return get_dataloader(batch_size=BATCH_SIZE, shuffle=True)

@pytest.fixture
def features_scaled(dataloader):
    all_data = []
    for batch in dataloader:
        all_data.append(batch)
    all_data = np.concatenate(all_data, axis=0)
    return all_data

def test_kmeans_init(mock_task):
    kmeans = KMeansClusterer(task=mock_task)
    assert hasattr(kmeans, 'max_clusters')
    assert kmeans.max_clusters > 0

def test_kmeans_run(features_scaled, mock_task):
    kmeans = KMeansClusterer(task=mock_task)
    results = kmeans.run(None, features_scaled)
    
    assert 'scores' in results
    assert isinstance(results['scores'], dict)
    assert 'Silhouette Score' in results['scores']
    assert 'Calinski-Harabasz Index' in results['scores']
    assert 'Davies-Bouldin Index' in results['scores']
    
    assert 'optimal_k' in results
    assert isinstance(results['optimal_k'], int)
    assert results['optimal_k'] > 0

def test_kmeans_find_elbow(mock_task):
    kmeans = KMeansClusterer(task=mock_task)
    k_values = range(2, 11)
    inertias = [100, 80, 60, 50, 45, 42, 40, 39, 38]
    elbow = kmeans.find_elbow(k_values, inertias)
    assert elbow in k_values
