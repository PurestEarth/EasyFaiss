"""Tests for utils
"""
from utils.cluster_utils import get_top_count,\
    most_common_for_cluster, get_cluster_names
from utils.string_utils import soft_set, remove_emoji, remove_chars,\
    get_word_trigrams, get_word_bigrams


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


def test_soft_set():
    """Tests
    """
    inputs = [1, 1, 2, 1, 2]
    assert soft_set(inputs) == [1, 2]
    inputs = [1, 1, 2, 1, 2, 6, 5, 5, 6, 4]
    assert soft_set(inputs) == [1, 2, 6, 5, 4]
    inputs = ['a', 'c', 'c', 'b', 'c']
    assert soft_set(inputs) == ['a', 'c', 'b']


def test_remove_emoji():
    """Tests
    """
    assert remove_emoji("Test✍🌷📌👈🏻🖥") == "Test"
    assert remove_emoji("🤔 Test 🙈 Test 😌") == " Test  Test "


def test_remove_chars():
    """Tests
    """
    assert remove_chars("T<><??e=s()t") == "Test"
    assert remove_chars("T<><??e&s()t", [
        '?', '<', '>', '&']) == "Tes()t"


def test_get_cluster_names():
    """Tests
    """
    clusters = [1, 2, 3, 1, 2, 3]
    sentences = ['Kitchen salmon', 'Two thirds of a row',
                 'Python java swift', 'Salmon with kitchen',
                 "Two scripts written in Java", "Python test"]
    cluster_names = get_cluster_names(clusters, sentences)
    assert cluster_names[1] == 'kitchen_salmon'
    assert cluster_names[2] == 'two_thirds'
    assert cluster_names[3] == 'python_java'


def test_n_grams():
    """Tests ngrams
    """
    sentence = "Kitchen salmon throws two thirds in a row"
    assert get_word_bigrams(sentence) == [('Kitchen', 'salmon'),
                                          ('salmon', 'throws'),
                                          ('throws', 'two'),
                                          ('two', 'thirds'),
                                          ('thirds', 'in'),
                                          ('in', 'a'),
                                          ('a', 'row')]
    assert get_word_trigrams(sentence) == [('Kitchen', 'salmon', 'throws'),
                                           ('salmon', 'throws', 'two'),
                                           ('throws', 'two', 'thirds'),
                                           ('two', 'thirds', 'in'),
                                           ('thirds', 'in', 'a'),
                                           ('in', 'a', 'row')]

    sentence = "Two"
    assert get_word_bigrams(sentence) == []
    assert get_word_trigrams(sentence) == []
    sentence = "Two thirds"
    assert get_word_bigrams(sentence) == [('Two', 'thirds')]
    assert get_word_trigrams(sentence) == []
