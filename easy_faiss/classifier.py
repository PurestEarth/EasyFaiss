"""Given index and classes of vectors one can use faiss index as classifier
"""
from typing import List
from collections import Counter
import torch


class Classifier:
    """_summary_
    """
    def __init__(self, index, classes: List) -> None:
        assert index.ntotal == len(classes)
        self.index = index
        self.classes = classes

    def classify(
            self, query: torch.Tensor, k: int = 1, min_threshold: float = 0.2):
        """Uses index and list of classes to assign class based on either
        exact match ( smaller than min_threshold ) or voting system
        Voting system returns most common label for neighbours or -1 if count
        is smaller than k/2

        Args:
            query (torch.Tensor): Query tensor
            k (int, optional): number of nearest neighbours. Defaults to 1.
            min_threshold (float, optional): minimal distance to consider
            vector an exact match. Defaults to 0.2.

        Returns:
            _type_: Class for each vector in query
        """
        classes = []
        distances_arr, index_arr = self.index.search(query, k)
        for indexes, distances in zip(index_arr, distances_arr):
            if distances[0] <= min_threshold:
                classes.append(self.classes[indexes[0]])
                continue
            top_count = Counter(list(
                map(lambda x: self.classes[x], indexes))).most_common(1)[0]
            if top_count[1] > k/2:
                classes.append(top_count[0])
            else:
                classes.append(-1)
        return classes

    def update_classes(self, classes: List) -> None:
        """_summary_

        Args:
            classes (List): new classes
        """
        assert self.index.ntotal == len(classes)
        self.classes = classes
