# RAG Tester

This is a basic UI app that will show the answer of a query by a selected model. The intention behind this is to
test performance of the model when changing different parameters. 

## Installation

1. **Clone the repository** into the desired directory.
    
    ```bash
    git clone https://github.com/jotarretx/RAG_Tester.git
    ```


2. **Install the required packages** in the requirements.txt file. Please note that there might be issues with llama.cpp and GPU usage, in that case, install that package separately.

    ```bash
    pip install -r requirements.txt
    ```
Current version uses llama.cpp, therefore **GGUF models are required**. For test purposes,
[Mistral 7b Instruct v0.2](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) and [Llama 2 13b Chat](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF) provided by TheBloke were used. Please download models **BEFORE** running the app and store them in the 
`models` folder.

**IMPORTANT**: To be able to load indexes, first they must be generated and stored. The `storage_generation` notebook under the `notebooks` folder can be used to generate the indexes and store them in the `storage` folder. The notebook uses the [BGE base embedding model](https://huggingface.co/BAAI/bge-base-en-v1.5), but feel free to try with others.

## Usage
1. Run the app
    ```bash
    python app.py
    ```

2. Select the model you want to use (A command prompt will show you available models from your models folder) by using the corresponding number appearing in the list.

3. Wait for the Gradio app to deploy and then open it in your browser.

4. Select your desired parameters and ask your question!

## Logs
Manual logs in the form of a single .json file has been created and stored in the `log` folder. 

