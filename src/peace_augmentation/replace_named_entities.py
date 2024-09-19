from typing import Dict, List

import torch
from flair.data import Sentence
from flair.embeddings import DocumentPoolEmbeddings
from scipy import spatial


def get_word_vector(word: str, doc_emb: DocumentPoolEmbeddings) -> torch.Tensor:
    """
    Given a word and a flair embedding returns a vector that represents it.
    """
    word = Sentence(word, use_tokenizer=True)
    doc_emb.embed(word)
    return word.embedding.to("cpu")


def find_closest_embeddings(vector_list: Dict[str, torch.Tensor], v: torch.Tensor) -> List[str]:
    """
    Given a vector `v` and a dictionary `vector_list` where keys are words and
    the values are vectors, return the `vector_list` keys (words) in descending
    order.

    One word w1 in `vector_list.keys()` is lower than w2 in `vector_list.keys()`
    if and only if the representation vector of w1, vector_list[w1], is closer
    in euclidean distance to `v` than the representation of w2 is to `v`.
    """
    return sorted(
        vector_list.keys(),
        key=lambda word: spatial.distance.euclidean(vector_list[word], v),
    )
