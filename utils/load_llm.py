from llama_index import ServiceContext, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_utils import messages_to_prompt
from utils import load_llamaCPP_model, MODEL_PATH, EMBEDDING_MODEL, SEMANTIC_CHUNKER


def load_model(model_path: str = MODEL_PATH, embedding_model: str = EMBEDDING_MODEL):
    """
    Loads a LLM from a local path, as form of an index with already loaded documents.

    Args:
        model_path (str): The path to the model to load. Defaults to the model path in the config file.
        embedding_model (str): The path to the embedding model to load. Defaults to the embedding model path in the config file.
    
    Returns:
        loaded_index (LlamaIndex): The loaded LLM index.
    """
    # Get path to the selected model
    model = model_path.split("/")[1]
    model_tag = model.split("-")[0]

    if model_tag == "mistral":
        global messages_to_prompt

        # Changing current messages_to_prompt function to a new one that works with Mistral based models (adapt as needed)
        def messages_to_prompt_mistral(messages):
            prompt = ""
            for message in messages:
                if message.role == "system":
                    prompt += f"<|system|>\n{message.content}</s>\n"
                elif message.role == "user":
                    prompt += f"<|user|>\n{message.content}</s>\n"
                elif message.role == "assistant":
                    prompt += f"<|assistant|>\n{message.content}</s>\n"

            # ensure we start with a system prompt, insert blank if needed
            if not prompt.startswith("<|system|>\n"):
                prompt = "<|system|>\n</s>\n" + prompt

            # add final assistant prompt
            prompt = prompt + "<|assistant|>\n"

            return prompt

        messages_to_prompt = messages_to_prompt_mistral

    # Load LLM
    llm = load_llamaCPP_model(model_path, messages_to_prompt)

    # Selecting Embedding model. Currently manually selected.
    embedding = embedding_model
    # Checking if Semantic Chunker is enabled
    embedding_tag = f"{embedding.split('/')[1]}_semantic_chunker" if SEMANTIC_CHUNKER else embedding.split("/")[1]
    
    embed_model = HuggingFaceEmbedding(embedding, max_length=512)

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        chunk_size=512,
        chunk_overlap=125,
    )

    # rebuild storage context
    storage_context = StorageContext.from_defaults(
        persist_dir=f"storage/{embedding_tag}"
    )

    # load index
    loaded_index = load_index_from_storage(
        storage_context, service_context=service_context
    )

    return loaded_index
