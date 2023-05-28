""" Faiss for clustering
    """
from typing import Union, Any
import faiss
import torch


class Clusterer:
    """_summary_
    """
    def __init__(
            self, dataset: torch.Tensor,
            index: Union[Any, None] = None,
            ncentroids: Union[int, None] = None,
            embed_dim: int = 768) -> None:
        if ncentroids is None:
            ncentroids = len(dataset)//1000
        if index is None:
            self.kmeans = faiss.Kmeans(
                embed_dim, ncentroids, niter=20, verbose=True,
                gpu=torch.cuda.is_available())
            self.kmeans.train(dataset)
            self.index = self.kmeans.index
        else:
            self.index = index
            if self.index.is_trained:
                self.index.add(dataset)
            else:
                self.index.train(dataset)

    def assign_clusters(
            self, query: torch.Tensor, k: int = 1, threshold: float = 0.7):
        """Attempts to assign clusters to given query, if
        distance is smaller than threshold returns -1

        Args:
            query (torch.Tensor): Query tensor
            k (int): number of nearest neighbours. Defaults to 1
            threshold (float, optional): Defaults to 0.7.

        Returns:
            List: Array of labels
        """
        distances_arr, labels_arr = self.index.search(query, k)
        return [
            -1 if distances[0] > threshold else labels[
                0] for distances, labels in zip(distances_arr, labels_arr)]
