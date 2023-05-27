"""Class including methods supporting clustering
"""

# take array of strings and array of classes, pick most popular words
from collections import Counter
from typing import List, Dict, Any


def get_top_count(lst: List[str], cutoff: int) -> List[str]:
    """returns top count of given array

    Args:
        lst (List[str]): list of words to count
        cutoff (int): number of most common cases to return

    Returns:
        _type_: _description_
    """
    counts = Counter(lst)
    return list(map(lambda x: x[0], counts.most_common(cutoff)))


def most_common_for_cluster(
        clusters, sentences, top_words=5) -> Dict[Any, List[str]]:
    """Returns most common words for given clusters

    Args:
        clusters (_type_): List of clusters
        sentences (_type_): List of sentences
        top_words (int, optional): How many of most common words to return.
        Defaults to 5.
        # TODO remove overlap
    Returns:
        Dict[Any, List[str]]: _description_
    """
    assert len(clusters) == len(sentences)
    cluster_to_acc = {}
    cluster_to_top_words = {}
    for cluster, sentence in zip(clusters, sentences):
        if cluster in cluster_to_acc:
            cluster_to_acc[cluster].extend(list(filter(lambda x: len(x) > 3,
                                                       sentence.split())))
        else:
            cluster_to_acc[cluster] = list(filter(lambda x: len(x) > 3,
                                                  sentence.split()))
    for cluster, acc in cluster_to_acc.items():
        cluster_to_top_words[cluster] = get_top_count(acc, top_words)
    return cluster_to_top_words
