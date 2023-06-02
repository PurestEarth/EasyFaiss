"""Test Classifier functionality
"""
import torch
from easy_faiss.easy_faiss import EasyFaiss
from easy_faiss.classifier import Classifier


def test_classify():
    """Test process of classification
    """
    dataset = torch.load('tests/data/test_tensor.pt')
    easy = EasyFaiss(dataset, 768, factory_string="Flat")
    _, index_arr = easy.search(dataset[:5], 1)
    assert list(map(lambda x: x[0], index_arr)) == [0, 1, 2, 3, 4]
    classes = ['test'] * easy.index.ntotal
    classifier = Classifier(easy.index, classes)
    assert classifier.classify(dataset[:5], 5) == [
        'test', 'test', 'test', 'test', 'test']
    assert classifier.classify(dataset[:2], 5) == [
        'test', 'test']
    classes[0] = 'test1'
    classes[1] = 'test2'
    classes[2] = 'test2'
    classes[2] = 'test2'
    classifier = Classifier(easy.index, classes)
    assert classifier.classify(dataset[:5], 5) == [
        'test1', 'test2', 'test2', 'test', 'test']
    assert classifier.classify(dataset[5:10], 5, 0.0) == [
        'test', 'test', 'test', 'test', 'test']
    easy = EasyFaiss(dataset[:-5], 768, factory_string="Flat")
    classes[0] = 'test1'
    classes[-1] = 'test2'
    classes[-2] = 'test2'
    classes[-32] = 'test2'
    classifier = Classifier(easy.index, classes[:-5])
    assert classifier.classify(dataset[-5:], 5) == [
        'test', 'test', 'test', 'test', 'test']
    classes = range(0, easy.index.ntotal)
    classifier = Classifier(easy.index, classes)
    # 2 inputs trigger threshold as exact matches
    assert classifier.classify(dataset[-5:], 5) == [1477, -1, -1, 1530, -1]
    assert classifier.classify(dataset[-5:], 5, 0) == [-1, -1, -1, -1, -1]


def test_test():
    """Test test function
    """
    dataset = torch.load('tests/data/test_tensor.pt')
    easy = EasyFaiss(dataset, 768, factory_string="Flat")
    classes = ['test'] * easy.index.ntotal
    classifier = Classifier(easy.index, classes)
    assert classifier.test(dataset[-5:], [
        'test', 'test', 'test', 'test', 'test']) == 1.0
    assert classifier.test(dataset[-5:], [
        'test', 'test', 'test', 'wrong', 'wrong']) == 0.6
