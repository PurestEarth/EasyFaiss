"""Class including methods supporting clustering
"""
from collections import Counter
from typing import List, Dict, Any
import torch
import random
from sklearn.manifold._t_sne import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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
        clusters: List[Any],
        sentences: List[str], top_words: int = 5) -> Dict[Any, List[str]]:
    """Returns most common words for given clusters

    Args:
        clusters (List[Any]): List of clusters
        sentences (List[str]): List of sentences
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


def tsne_visualization(clusters: List[Any],
                       embeddings: torch.Tensor, n_components: int = 3,
                       show: bool = True, zero_label: Any = -1) -> TSNE:
    """

    Args:
        clusters (List[Any]): Clusters
        embeddings (torch.Tensor): Embeddings to group
        n_components (int, optional): Dimension of vector. Defaults to 3.
        show (bool, optional): if True scatter plot will be shown.
        Defaults to True.

    Returns:
        _type_: TSNE model
    """
    model = TSNE(n_components=n_components, random_state=1)
    model.fit_transform(embeddings)
    if show:
        plt.figure(figsize=(50, 50))
        plt.scatter(model.embedding_[:, 0], model.embedding_[:, 1])
        non_grey = [ele for ele in list(
            mcolors.CSS4_COLORS.keys()) if ele not in [
                'gray', 'grey', 'lightgrey', 'lightgray']]
        label_to_color = {}
        for i, cluster in enumerate(clusters):
            if len(clusters) < len(mcolors.CSS4_COLORS.items()):
                label_to_color[cluster] = mcolors.CSS4_COLORS[non_grey[i]]
            else:
                label_to_color[cluster] = mcolors.CSS4_COLORS[
                    random.choice(non_grey)]
        label_to_color[zero_label] = 'grey'
        for embed, label in zip(model.embedding_, clusters):
            plt.scatter(embed[0], embed[1], c=label_to_color[label],
                        label=label)

        plt.show()
    return model
