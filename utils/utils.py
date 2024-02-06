# Utilities for loading LLM

from llama_index.retrievers import BaseRetriever
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import completion_to_prompt

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
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
        # You can pass in the URL to a GGML model to download it automatically
        # model_url=model_url,
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path=model_path,
        temperature=0.2,
        max_new_tokens=1000,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=3900,
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