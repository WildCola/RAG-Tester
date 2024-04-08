from llama_index import get_response_synthesizer, PromptTemplate
from llama_index.query_engine import RetrieverQueryEngine
from utils import QA_PROMPT

def initialise_response_synthesizer(retriever):
    """
    Initialises the response synthesizer using a retriever.

    Args:
        retriever (HybridRetriever): The retriever to use for obtaining relevant context.
    
    Returns:
        synth (BaseSynthesizer): The response synthesizer.
    """

    service_context = retriever.get_service_context()
    qa_prompt_tmpl = PromptTemplate(QA_PROMPT)

    synth = get_response_synthesizer(
        text_qa_template=qa_prompt_tmpl,
        service_context=service_context,
    )
    
    return synth


def initialise_query_engine(retriever, synth, reranker: list):
    """
    Initialises the query engine using a retriever and response synthesizer.
    
    Args:
        retriever (HybridRetriever): The retriever to use for obtaining relevant context.
        synth (BaseSynthesizer): The response synthesizer to use for generating responses.
        reranker (list): The list of rerankers (node_postprocessors) to use for reranking the responses.
        
    Returns:
        query_engine: The full query engine for RAG.
    """
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=reranker,
        response_synthesizer=synth,
    )
    
    return query_engine