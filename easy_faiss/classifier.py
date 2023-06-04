"""Given index and classes of vectors one can use faiss index as classifier
"""
from typing import List
from collections import Counter
from easy_faiss.easy_faiss import EasyFaiss
import torch
from sklearn.metrics import classification_report


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

    def test(self, test_ds: torch.Tensor, true_labels: List, k: int = 5):
        """Given test ds returns accuracy of classifier

        Args:
            test_ds (torch.Tensor): Test DS
            true_labels (List): labels of given test ds

        Returns:
            float:Accuracy score for given dataset
        """
        return self.compare_labels(true_labels, self.classify(test_ds, k))

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

    def classification_metrics(self,
                               test_ds: torch.Tensor,
                               true_labels: List,
                               k: int = 5) -> str:
        """Returns classification metrics for given test dataset

        Args:
            test_ds (torch.Tensor): Test DS
            true_labels (List): labels of given test ds

        Returns:
            str: classification metrics
        """
        return classification_report(true_labels, self.classify(test_ds, k=5))

    def prune(self, dataset: torch.Tensor,
              threshold=0.2, k=5,
              return_indexes=False,
              factory_string=None):
        """In specific use case, using classification with k=1,
        it is possible to get rid of certain portion of dataset and
        index by removing nearest neighbours. Might lower overall precision.

        Args:
            dataset (torch.Tensor): Has to be the same dataset index was
                trained on
            threshold (float, optional): Similarity threshold. Defaults to 0.2.
            k (int, optional): Num of nearest neighbours. Defaults to 5.
            return_indexes (bool, optional): If indexes to remove should be returned. Defaults to False.
            factory_string (_type_, optional): If indexes arent returned Classifier will prune redundant data. Defaults to None.

        Returns:
            _type_: _description_
        """
        init_class_length = len(self.classes)
        distances_arr, index_arr = self.index.search(dataset, k)
        closest_index_arr = []
        for i, (distances, indexes) in enumerate(
                zip(distances_arr, index_arr)):
            closest = [index for distance, index in zip(
                distances, indexes) if index != i and distance < threshold and
                       self.classes[index] == self.classes[i]]
            closest_index_arr.append(closest)
        necessary, to_remove = [], []
        for i, closest in enumerate(closest_index_arr):
            if len(closest) > 0 and i not in necessary:
                for neighbour in closest:
                    if neighbour not in to_remove:
                        necessary.append(neighbour)
                        to_remove.append(i)
                        break
        if return_indexes:
            return to_remove
        mask = torch.ones_like(dataset, dtype=torch.bool)
        mask[to_remove] = False
        dataset = torch.masked_select(dataset, mask=mask)
        dataset = torch.reshape(
            dataset, (init_class_length-len(to_remove), 768))
        self.index = EasyFaiss(
            dataset, dataset.size()[-1], factory_string=factory_string)
        for index in sorted(to_remove, reverse=True):
            del self.classes[index]
        assert len(self.classes) == dataset.size()[0]
