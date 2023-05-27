"""Tests for utils
"""
from utils.cluster_utils import get_top_count, most_common_for_cluster


def test_get_top_count():
    """Tests get top count
    """
    test_input = ["lambda", "lambda", "social", "social",
                  "lambda", "lambda", "three", "test"]
    top_count = get_top_count(test_input, 2)
    assert top_count == ['lambda', 'social']
    top_count = get_top_count(test_input, 5)
    assert top_count == ['lambda', 'social', 'three', 'test']
    test_input.extend(["lambda", "test", "1", "2", "3", "4"])
    top_count = get_top_count(test_input, 5)
    assert top_count == ['lambda', 'social', 'test', 'three', '1']


def test_most_common_for_cluster():
    """Tests
    """
    sentences = ["One two lambda", "One two lambda", "Fish on a stick",
                 "One three sigma", "Fish in the stick sea"]
    clusters = [1, 1, 2, 1, 2]
    most_common = most_common_for_cluster(clusters, sentences)
    assert most_common[1] == ['lambda', 'three', 'sigma']
    assert most_common[2] == ['Fish', 'stick']
    most_common = most_common_for_cluster(clusters, sentences, 1)
    assert most_common[1] == ['lambda']
    assert most_common[2] == ['Fish']
