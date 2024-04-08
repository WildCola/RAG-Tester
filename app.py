import gradio as gr
import subprocess
from datetime import datetime
from retrievers import initialise_retrievers
from postprocessors import initialise_postprocessors
from query_engine import initialise_response_synthesizer, initialise_query_engine
from utils import get_source_and_context, load_model, log_output, EMPTY_RESPONSE

# Running ngrok
subprocess.Popen(["ngrok", "http", "--domain=directly-uncommon-sunbird.ngrok-free.app", "7860"])

# Load the model
index = load_model()

# Initialize retrievers
hybrid_retriever = initialise_retrievers(index)

# Configure rerankers (obtains list of )
reranker = initialise_postprocessors()

# Initialise synthesizer
synth = initialise_response_synthesizer(hybrid_retriever)

# Initialise query engine
query_engine = initialise_query_engine(hybrid_retriever, synth, reranker)


def process_inputs(question):
    """
    Process the inputs and return the answer, source, and context.

    Args:
    question (str): The question to be answered.

    Returns:
    answer (str): The answer to the question.
    source (str): The source of the answer.
    context (str): The context used to create the answer.
    """
    initial_time = datetime.now()
    response = query_engine.query(question)

    source, context = get_source_and_context(response)
    answer = (
        response.response if response.response != "Empty Response" else EMPTY_RESPONSE
    )

    # Logging
    log_output(question, answer, source, context, initial_time)

    return answer, source, context


iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Textbox(lines=2, placeholder="Make your question here...", label="Question")
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
    allow_flagging="never",
)

iface.queue()
iface.launch()    # share=True changed for ngrok