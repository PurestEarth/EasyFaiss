""" Tests basic clustering functionality
"""
import torch
from easy_faiss.easy_faiss import EasyFaiss


def test_assign_clusters() -> None:
    """Test valid outputs
    """
    dataset = torch.load('tests/data/test_tensor.pt')
    easy = EasyFaiss(dataset, 768)
    _, index_arr = easy.search(dataset[:5], 1)
    assert list(map(lambda x: x[0], index_arr)) == [0, 1, 2, 3, 4]
    _, index_arr = easy.search(dataset[:5], 5)
    assert list(map(lambda x: x[0], index_arr)) == [0, 1, 2, 3, 4]
    easy = EasyFaiss(dataset, 768, factory_string="Flat")
    distances_arr, index_arr = easy.search(dataset[:5], 1)
    assert list(map(lambda x: x[0], index_arr)) == [0, 1, 2, 3, 4]
    assert list(map(lambda x: x[0], distances_arr)) == [0, 0, 0, 0, 0]
    easy = EasyFaiss(dataset=dataset, factory_string="SQ8")
    distances_arr, index_arr = easy.search(dataset[:5], 1)
    assert list(map(lambda x: x[0], index_arr)) == [0, 1, 2, 3, 4]
