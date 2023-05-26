""" Tests basic clustering functionality
"""
import torch
from easy_faiss.clusterer import Clusterer


def test_assign_clusters() -> None:
    """Test valid outputs
    """
    dataset = torch.load('tests/data/test_tensor.pt')
    clusterer = Clusterer(dataset=dataset, ncentroids=5)
    assert clusterer.index.is_trained
    assert clusterer.assign_clusters(dataset[:5]) == [3, 3, -1, 2, -1]
    assert clusterer.assign_clusters(
        dataset[:5], threshold=0.5) == [3, 3, -1, 2, -1]
    assert clusterer.assign_clusters(
        dataset[:5], threshold=0.3) == [-1, -1, -1, 2, -1]
