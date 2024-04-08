import os
from llama_index import (
    ServiceContext,
    StorageContext, 
    load_index_from_storage
)
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms.llama_utils import messages_to_prompt
from utils import load_llamaCPP_model

def load_model():
    """
    Loads a LLM from a local path, as form of an index with already loaded documents.
    
    Returns:
        loaded_index (LlamaIndex): The loaded LLM index.
    """
    # Loading models paths
    models_path = "models"
    models = [f for f in os.listdir(models_path) if os.path.isfile(os.path.join(models_path, f))]
    try:
        # remove .gitignore by specifying the name
        models.remove(".gitignore")
    except:
        pass
    
    try:
        # remove anything ending with Zone.Identifier
        models = [m for m in models if not m.endswith("Zone.Identifier")]
    except:
        pass
    
    # From every entry, remove everything after the first dot
    print("********************")
    print("Available models:")
    for i, m in enumerate(models):
        print(f"{i}: {m.split('.')[0]}")
    print("********************")

    # Select a model. The user can only input a number between 0 and len(models)-1, if he inputs something else, the program will ask again
    while True:
        try:
            model_index = int(input("\nSelect a model: "))
            if model_index >= 0 and model_index < len(models):
                break
            else:
                print("Invalid input. Please enter a number between 0 and " + str(len(models)-1) + " according to the selection shown above.")
        except ValueError:
            print("Invalid input. Please enter a number between 0 and " + str(len(models)-1) + " according to the selection shown above.")

    # Get path to the selected model
    model_path = os.path.join(models_path, models[model_index])
    model_tag = models[model_index].split('-')[0]

    if model_tag != "llama":
        global messages_to_prompt
        # Changing current messages_to_prompt function to a new one that works with Mistral based models
        def messages_to_prompt_mistral(messages):
            prompt = ""
            for message in messages:
                if message.role == 'system':
                    prompt += f"<|system|>\n{message.content}</s>\n"
                elif message.role == 'user':
                    prompt += f"<|user|>\n{message.content}</s>\n"
                elif message.role == 'assistant':
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
    embedding = "BAAI/bge-base-en-v1.5"   # BAAI/bge-base-en-v1.5   BAAI/bge-large-en-v1.5
    embedding_tag = embedding.split('/')[1]
    embed_model = HuggingFaceEmbedding(embedding, max_length=512)
        
    service_context = ServiceContext.from_defaults(
        llm=llm, 
        embed_model= embed_model,
        chunk_size=512,
        chunk_overlap=125,
    )

    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=f"storage/{embedding_tag}")

    # load index
    loaded_index = load_index_from_storage(storage_context, service_context= service_context)
    
    return loaded_index


