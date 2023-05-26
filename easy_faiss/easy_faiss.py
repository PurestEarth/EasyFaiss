"""Wrapper for faiss."""
from typing import Union, Tuple, Any
import faiss
import torch


class EasyFaiss:
    """
        Class extending index for typical use
    """
    def __init__(
            self,
            dataset: torch.Tensor,
            dim: Union[int, None] = None,
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
            # self.index = faiss.IndexFlatL2(dim)
        if self.index.is_trained:
            self.index.add(dataset)
        else:
            self.index.train(dataset)
            self.index.add(dataset)
        
    def get_index(self) -> Any:
        """
        Returns:
            _: faiss index
        """
        return self.index

    def add(self, data: torch.Tensor) -> None:
        """add more data to index

        Args:
            data (torch.Tensor): data to add to index
        """
        self.index.add(data)

    def search(
            self, vectors: torch.Tensor, k: int
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """query index

        Args:
            vector (torch.Tensor): tensor of vectors to query
            k (int): number of neighbours to return

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: D, I - Distance and Index
        PQ Index might return nothing if vector it
        was trained has less than ~1000 examples
        Flat Index advised for small datasets
        """
        return self.index.search(vectors, k)
