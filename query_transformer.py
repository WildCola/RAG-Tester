"""Query Transformer

This script tests different query transformation methods to improve the retrieval efficiency of the RAG model.
It mainly focuses on query rewriting using custom methods, HyDE Query Transform, and the generation of sub-queries.

"""


# from llama_index import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.indices.query.query_transform import HyDEQueryTransform
from llama_index.query_engine import TransformQueryEngine
from llama_index.question_gen.llm_generators import LLMQuestionGenerator
# from retrievers import initialise_retrievers
# from postprocessors import initialise_postprocessors
# from query_engine import initialise_response_synthesizer, initialise_query_engine
# from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt
# from storage_generation import service_context, storage_context
from utils import load_llamaCPP_model, load_model


model = load_llamaCPP_model("models/zephyr-7b-beta.Q5_K_M.gguf", messages_to_prompt=messages_to_prompt)

# hybrid_retriever = initialise_retrievers(model)

# Configure rerankers (obtains list of )
# reranker = initialise_postprocessors()

# Initialise synthesizer
# synth = initialise_response_synthesizer(hybrid_retriever)

# Initialise query engine
# query_engine = initialise_query_engine(hybrid_retriever, synth, reranker)


def generate_custom_queries(llm, query: str, num_queries: int = 3) -> str:
    """Generate custom queries by using a local LLM.
    
    llm (LlamaCPP): LLM for query transformation
    query (str): custom user query input
    num_queries (int): number of queries to be generated during transformation process (defaults to 3)
    
    Returns:
        transformed_queries (str): newly generated queries by LLM based on original query
    """
    prompt = f"""You are a helpful assistant that generates multiple search queries based on a \
        single input query. Generate {num_queries} search queries, one on each line, \
        related to the following input query:
        Query: {query}"""
    response = llm.complete(prompt)
    transformed_queries = f"Original query: {query}\nGenerated queries: {response.text}"
    return transformed_queries

# query_transformation = generate_custom_queries(model, "What is the EU AI act about?")
# print(query_transformation)


def generate_hyde_answers(query):
    # TODO: solve ValueError: Could not load OpenAI model
    index = load_model()
    query_engine = index.as_query_engine()
    hyde = HyDEQueryTransform(include_original=True)
    hyde_query_engine = TransformQueryEngine(query_engine=query_engine, query_transform=hyde)
    response = hyde_query_engine.query(query)
    return response

# print(generate_hyde_answers("What is the European Parliament?"))


def generate_sub_queries(llm, query):
    query_generator = LLMQuestionGenerator(llm_predictor=llm, prompt=query)
    sub_queries = query_generator.generate(tools=[], query=query)
    return sub_queries  

print(generate_sub_queries(model, "What is the role of the Secretary General of the European Parliament?"))
