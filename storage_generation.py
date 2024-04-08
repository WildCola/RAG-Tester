import os
import warnings
import pickle
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader, 
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from utils import (
    load_llamaCPP_model,
    MODEL_PATH,
    EMBEDDING_MODEL,
)


warnings.filterwarnings("ignore")
print("Loading model")
# Get path to the selected model
model_path = MODEL_PATH
model_tag = model_path.split("/")[1].split('-')[0]

if not model_tag.startswith("llama"):
        # The following prompt works well with Mistral
        def messages_to_prompt(messages):
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

llm = load_llamaCPP_model(model_path, messages_to_prompt)

# Selecting Embeddings model
embedding = EMBEDDING_MODEL
embedding_tag = embedding.split('/')[1]
embed_model = HuggingFaceEmbedding(embedding, max_length=512)

service_context = ServiceContext.from_defaults(
    llm=llm, 
    embed_model= embed_model, 
    chunk_size=512,
    chunk_overlap=125,
)
# Saving data path
data_path = 'data'

# First check if user wants new embeddings or to add documents
input_mode = ''
while input_mode not in ['new', 'add']:
    input_mode = input("Do you want to generate new embeddings or add documents to an existing index? (new/add): ").lower()

docstore = ''
if input_mode == 'new':
    while docstore not in ['new', 'load']:
        docstore = input("Do you want to create a new document store or load an existing one? (new/load): ").lower()

    if docstore == 'new':    
        # Data ingestion
        documents = SimpleDirectoryReader(data_path, exclude_hidden=True).load_data()

        # Storing documents as a list to avoid loading them again
        with open('storage/documents/documents.pickle', 'wb') as f:
            pickle.dump(documents, f)
        print(f"Stored {len(documents)} document nodes.")
    else:
        # Opening the stored documents
        with open('storage/documents/documents.pickle', 'rb') as f:
            documents = pickle.load(f)
        print(f"Loaded {len(documents)} document nodes.")

    # Creating embeddings
    vector_index = VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True)

    # Persisting embeddings
    vector_index.storage_context.persist(persist_dir=f"storage/{embedding_tag}")
    print("Embeddings generated and saved to disk.")
else:
    print("\nLoading existing index...")
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=f"storage/{embedding_tag}")

    # load index
    vector_index = load_index_from_storage(storage_context, service_context=service_context)
    
    doc_path = input("Please enter the document NAME (e.g. EUWhoiswho_EP_EN.pdf): ")
    # Adding new documents to existing index
    file_path = os.path.join(data_path, doc_path)

    # Data ingestion
    try:
        new_documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    except:
        print("Error: File not found.")
        exit()

    # Add to index
    for chunk in new_documents:
        vector_index.insert(chunk, show_progress=True)

    # Persist to disk
    vector_index.storage_context.persist(persist_dir=f"storage/{embedding_tag}")
    print(f"Document {doc_path} added to index.")
    input("\nPress Enter to exit...")

# Remember to update the document store in case it is needed in the future! (Loading documents section)


