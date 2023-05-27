"""Hugging Face Models used as means to vectorize string inputs
"""
from typing import Union, List
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm


class HFAutoModel:
    """Class providing vectorizing functionality of HF model
    """
    def __init__(self,
                 tokenizer: Union[str, AutoTokenizer] = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                 model: Union[str, AutoModel] = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2') -> None:
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        if isinstance(model, str):
            self.model = AutoModel.from_pretrained(model)
        else:
            self.model = model


    def mean_pooling(
            self, model_output: transformers.utils.ModelOutput,
            attention_mask: torch.Tensor):
        """Takes attention mask into account and averages across tokens
            correct averaging
            batch_size, token_num, 768 -> batch_size, 768
        Args:
            model_output (transformers.utils.ModelOutput): HF model output
            attention_mask (torch.Tensor):

        Returns:
            torch.Tensor:
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()).float()
        return torch.sum(
            token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)
            
    def process(self, data: List[str], batch_size: int = 64,
                drop_remainder: bool = True):
        """Process data into dataset

        Args:
            data (List[str]): data to vectorize
            batch_size (int, optional): Batch Size. Defaults to 64.
            drop_remainder (bool, optional): If last batch has different size 
            than batch size it will be dropped if set to true.
            Defaults to True.

        Returns:
            torch.Tensor: Processed dataset
        """
        dataset = []
        batched_preprocessed = [data[i: i + batch_size] for i in range(
            0, len(data), batch_size)]
        if len(batched_preprocessed[-1]) != batch_size and drop_remainder:
            batched_preprocessed = batched_preprocessed[:-1]
        for batch in tqdm(batched_preprocessed):
            # Tokenize sentences

            encoded_input = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors='pt')
            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)

            # Perform pooling
            sentence_embeddings = self.mean_pooling(
                model_output, encoded_input['attention_mask'])

            # Normalize embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

            dataset.append(sentence_embeddings)
        dataset = torch.stack(dataset)
        dataset = torch.reshape(
            dataset, (batch_size * dataset.size()[0], 768))
        return dataset
