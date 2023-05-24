"""Wrapper for faiss."""
from typing import Union
import faiss
import torch


class EasyFaiss:
    """
        Class extending index for typical use
    """
    def __init__(
            self, dim: Union[int, None],
            dataset: torch.Tensor,
            factory_string: Union[str, None] = None) -> None:
        if dim is None:
            dim = dataset.size()[-1]
        if factory_string:
            self.index = faiss.index_factory(dim, factory_string)
        else:
            # defaults to PQ
            assert dim % 8 == 0
            nbits = min(len(dataset)//1000, 11)
            self.index = faiss.IndexPQ(dim, 8, nbits)
        if self.index.is_trained:
            self.index.add(dataset)
        else:
            self.index.train(dataset)
            self.index.add(dataset)

    def get_index(self):
        """
        Returns:
            _: faiss index
        """
        return self.index

    def add(self, data: torch.Tensor):
        """add more data to index

        Args:
            data (torch.Tensor): data to add to index
        """
        self.index.add(data)
