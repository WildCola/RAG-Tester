import streamlit as st
import html
from datetime import datetime
from retrievers import initialise_retrievers
from postprocessors import initialise_postprocessors
from query_engine import initialise_response_synthesizer, initialise_query_engine
from utils import get_source_and_context, load_model, log_output, EMPTY_RESPONSE



@st.cache_resource()
def load_resources():
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

    return query_engine

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
    response = query_engine.query(question)

    source, context = get_source_and_context(response)
    answer = (
        response.response if response.response != "Empty Response" else EMPTY_RESPONSE
    )

    return answer, source, context

st.set_page_config(layout="wide")
# Streamlit code
st.title('EP Assistant (PoC) v2.0')

# Load resources
query_engine = load_resources()

# User inputs question
question = st.text_input('Make your question here:')

if question:
    # Model generates answer
    initial_time = datetime.now()
    answer, source, context = process_inputs(question)
    
    # Store the answer in the session state
    st.session_state['answer'] = answer
    st.session_state['source'] = source
    st.session_state['context'] = context

    # Creating columns
    col1, col2, col3 = st.columns(3)

    # Display the answer in the left column
    if 'answer' in st.session_state:
        col1.write(f'Answer:\n{st.session_state["answer"]}')

        # Display the source, context, and feedback in the right column
        col2.write(f'Source:\n{st.session_state["source"]}')

        escaped_context = html.escape(st.session_state["context"])
        col2.text_area('Context:', key='context', height=500)
        
        
        # Initialize session state for Streamlit
        if 'feedback' not in st.session_state:
            st.session_state['feedback'] = None
        
        # User provides feedback
        feedback = col3.selectbox('How would you rate the answer?', ['Good', 'Neutral', 'Bad'], index=None, placeholder='Select an option')
        # Write under the feedback selection the importance of the feedback
        st.session_state['feedback'] = col3.write('Feedback is needed in order to save the result, please remember to provide it!')

        # Create a submit button that logs the question, answer, source, context, initial_time, and feedback
        if col3.button('Submit'):
            log_output(question, answer, source, context, initial_time, st.session_state['feedback'])
            del question
            st.session_state['feedback'] = None
        
        
