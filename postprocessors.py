from llama_index.postprocessor import SentenceTransformerRerank, SimilarityPostprocessor
from utils import SIMILARITY_CUTOFF, RERANKER, RERANKER_TOP_N


def initialise_postprocessors(top_n: int = RERANKER_TOP_N, model: str = RERANKER, similarity_cutoff: float = SIMILARITY_CUTOFF) -> list:
    """
    Initialises the reranker postprocessors.
    
    Args:
        top_n (int): The number of chunks to retrieve. Defaults to the top n in the config file.
        model (str): The sentence transformer model to use. Defaults to the reranker model in the config file.
        similarity_cutoff (float): The similarity cutoff to use. If the similarity score is below this threshold, the chunk is discarded. Defaults to the similarity cutoff in the config file.
        
    Returns:
        list: A list of postprocessors.
    """
    sentence_transformer_reranker = SentenceTransformerRerank(
        top_n=top_n, model=model
    )
    similarity_reranker = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
    
    return [sentence_transformer_reranker, similarity_reranker]