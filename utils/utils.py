# Utilities for loading LLM
import json
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_utils import completion_to_prompt
from datetime import datetime

# Constants
SIMILARITY_TOP_K = 15
RERANKER_TOP_N = 3
SIMILARITY_CUTOFF = 0.25
QA_PROMPT = """You are an expert on world politics and issues. You have access to a database of questions and answers from european parliamentaries so your specialties are European politics and European institutions such as the European Parliament (EP), the European Commission (EC) and the European Union (EU) as a whole.
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and NOT prior knowledge, answer the query.
Make the answers you give as detailed as possible, don't leave out any possible relevant information.
If you find conflicting context information prioritize the more recent one.
When mentioning events be sure to provide their dates.
NEVER use ambigous time descriptors such as 'now' or 'soon', ALWAYS ground your answers to a clear timeframe.
AVOID using phrases like 'Based on the given context', 'Based on the provided context', 'Based on the information provided',
'As an expert on' or similar.
Do not mention the pdf provided.

Query:
--------------------
{query_str}
---------------------"""
EMPTY_RESPONSE = "Apologies, given that the question is not related to the European Parliament or the context provided did not give enough information, I cannot answer that question."
MODEL_PATH = "models/mixtral-8x7b-instruct-v0.1.Q8_0.gguf" # "models/mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf"
EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2' # "intfloat/e5-large-v2" #   # "BAAI/bge-base-en-v1.5"    EuropeanParliament/eubert_embedding_v1
RERANKER = "BAAI/bge-reranker-base"
SEMANTIC_CHUNKER = False


def load_llamaCPP_model(model_path:str, messages_to_prompt: callable):
    """
    Load a LLM from a local path.
    
    Args:
        model_path (str): The path to the model to load. 
        messages_to_prompt (function): A function that transforms a list of messages into a specific model prompt.
        
    Returns:
        llm (LlamaCPP): The loaded LLM.
    """
    
    llm = LlamaCPP(
        # You can pass in the URL to a GGUF model to download it automatically
        # model_url=model_url,
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path=model_path,
        temperature=0.2,
        max_new_tokens=1000,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=4096,
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={"n_gpu_layers": -1},
        # transform inputs into specific model format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )
    
    return llm

def get_source_and_context(answer) -> str:
    """
    Obtain sources and context from generated response.
    
    Args:
        answer (Answer): The answer object which will be striped.
    
    Returns:
        consolidated_source (str): The source of the provided answer.
        consolidated_context (str): The context of the provided answer.        
    """
    source_dict = {}
    context = []
    include_document_info = False # Just a flag to include/exclude document info. This is given in source so for now flagged as False.
    if answer.source_nodes:
        for i, key in enumerate(answer.metadata):
            file_name = answer.metadata[key]['file_name']
            page_label = answer.metadata[key]['page_label']
            
            if file_name in source_dict:
                source_dict[file_name].add(page_label)
            else:
                source_dict[file_name] = {page_label}
            
            context.append((
        '\nDocument name: ' + answer.metadata[key]['file_name'] + ', page: ' + answer.metadata[key]['page_label'] if include_document_info else ''
    ) + f"""

    {answer.source_nodes[i].text}

    Similarity score: {answer.source_nodes[i].score}
    """
            )

        source = ""
        for file_name, page_labels in source_dict.items():
            sorted_page_labels = sorted(list(page_labels), key=int)  # Convert to list and sort as integers
            source += f"{file_name}, page(s): {', '.join(sorted_page_labels)}\n"
        
        consolidated_source = source.strip()
        consolidated_context = '\n'.join(context)
    

        return consolidated_source, consolidated_context
    else:
        return "No source found", "No context found"


def log_output(question:str, answer:str, source:str, context:str, initial_time:datetime, feedback:str=None):
    """
    Log the output of the Q&A with additional parameters.
    
    Args:
        question (str): The question asked.
        answer (str): The answer to the question.
        source (str): The source of the answer.
        context (str): The context used to create the answer.
        initial_time (datetime): The initial time of the request.
    """
    # Get the current date and time
    current_datetime = datetime.now()
    elapsed_time = current_datetime - initial_time

    # Log the output
    with open("log/output_log.json", "a") as f:
        json.dump(
            {
                "datetime": initial_time.isoformat(),
                "processing_time": str(elapsed_time),
                "k": RERANKER_TOP_N,
                "question": question,
                "answer": answer,
                "source": source,
                "context": context,
                "feedback": feedback if feedback else "No feedback given",
            },
            f,
        )
        f.write("\n")