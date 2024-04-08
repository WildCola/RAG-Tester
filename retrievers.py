from llama_index.retrievers import VectorIndexRetriever, BM25Retriever, BaseRetriever
from utils import SIMILARITY_TOP_K

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self._index = vector_retriever._index
        super().__init__()

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes

def initialise_retrievers(index):
    """
    Initialize the retrievers for the application.

    Args:
    index (VectorStoreIndex): The LlamaIndex to use for the retrievers.

    Returns:
    hybrid_retriever (HybridRetriever): The hybrid retriever to use for obtaining the most relevant context.
    """
    # Initialize retrievers
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=SIMILARITY_TOP_K,
    )
    bm25_retriever = BM25Retriever.from_defaults(
        index=index,
        similarity_top_k=SIMILARITY_TOP_K,
    )

    # Combining both approaches
    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)

    return hybrid_retriever



