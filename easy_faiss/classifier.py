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

    def test(self, test_ds: torch.Tensor, true_labels: List):
        """Given test ds returns accuracy of classifier

        Args:
            test_ds (torch.Tensor): Test DS
            true_labels (List): labels of given test ds

        Returns:
            float:Accuracy score for given dataset
        """
        print(self.classify(test_ds))
        return self.compare_labels(true_labels, self.classify(test_ds))

    def compare_labels(self, true_labels, predicted_labels):
        """Compares a list of true labels with a list of predicted labels.

        Args:
            true_labels (List[Any]): List of true labels.
            predicted_labels (List[Any]): List of predicted labels.

        Returns:
            float: Accuracy score, representing the percentage of correct 
            predictions.
        """
        correct_count = sum(
            1 for true, pred in zip(true_labels, predicted_labels
                                    ) if true == pred)
        total_count = len(true_labels)
        accuracy = correct_count / total_count
        return accuracy
