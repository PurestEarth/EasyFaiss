""" Tests basic clustering functionality
"""
from transformers import AutoTokenizer, AutoModel
from vectorizers.hf_automodel import HFAutoModel


def test_assign_clusters() -> None:
    """Test valid outputs
    """

    hf_model = HFAutoModel()
    sentences = ["Test sentence"] * 400
    dataset = hf_model.process(sentences)
    assert len(dataset) == 384
    tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    model = AutoModel.from_pretrained(
        'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    hf_model = HFAutoModel(tokenizer, model)
    dataset = hf_model.process(sentences, 16)
    assert len(dataset) == 400
