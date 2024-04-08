import gradio as gr
import json
from datetime import datetime
from llama_index import get_response_synthesizer
from llama_index.retrievers import VectorIndexRetriever, BM25Retriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.prompts import PromptTemplate
from llama_index.postprocessor import SentenceTransformerRerank
from utils import get_source_and_context, load_model, HybridRetriever


index = load_model()

def process_inputs(prompt, k, n, rerank, question):    
    similarity_top_k = k 
    reranker_top_k = n
    
    # configure retriever
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=similarity_top_k,
    )

    bm25_retriever = BM25Retriever.from_defaults(
        index=index,
        similarity_top_k=similarity_top_k,
    )
    
    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
    
    # configure reranker
    reranker = SentenceTransformerRerank(top_n=reranker_top_k, model="BAAI/bge-reranker-base")
    
    service_context = vector_retriever.get_service_context()
    
    qa_prompt_tmpl_str = prompt

    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
    
    synth = get_response_synthesizer(
        text_qa_template=qa_prompt_tmpl,
        service_context=service_context,
    )
    
    query_engine = RetrieverQueryEngine(
        retriever=hybrid_retriever,
        node_postprocessors=[reranker] if rerank else [],
        response_synthesizer=synth,
    )
    
    response = query_engine.query(question)
    
    source, context = get_source_and_context(response)
    answer = response.response
    
    # Logging
    # Get the current date and time
    current_datetime = datetime.now().isoformat()

    # Log the output
    with open('log/output_log.json', 'a') as f:
        json.dump({
            'datetime': current_datetime,
            'prompt': prompt,
            'retriever top k': similarity_top_k,
            'reranker top k': reranker_top_k,
            'reranker enabled': rerank,
            'question': question,
            'answer': answer,
            'source': source,
            'context': context,
        }, f)
        f.write('\n')

    return answer, source, context

iface = gr.Interface(
    fn=process_inputs, 
    inputs=[
        gr.Textbox(
            lines=2, 
            value="""Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Avoid using phrases like 'Based on the given context', 'Based on the provided context' or similar.
Query:
--------------------
{query_str}
---------------------""", 
            label="Prompt (DO NOT REMOVE {context_str} and {query_str})"
        ),
        gr.Slider(minimum=1, maximum=30, value=15, label="Similarity top k for context retrieval"),
        gr.Slider(minimum=1, maximum=10, value=3, label="Similarity top k for reranker"),
        gr.Checkbox(label="Enable reranker?", value=True),
        gr.Textbox(
            lines=2, 
            placeholder="Make your question here...", 
            label="Question"
        )
    ], 
    outputs=[
        gr.Textbox(
            label="Answer",
            autoscroll=False,
        ),
        gr.Textbox(
            label="Source",
            autoscroll=False,
        ),
        gr.Textbox(
            label="Context",
            autoscroll=False,
        ),
    ],
    allow_flagging='never',
)


iface.launch(share=True)