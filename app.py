import json
import os
import sys
import boto3
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

## We will be using Titan Embeddings Model To generate Embedding
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding And Vector Store
from langchain.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

## Data Ingestion Implementation
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=10000)
    docs = text_splitter.split_documents(documents)
    return docs

# Vector Embedding and Vector Store Implementation
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def get_mistral_llm():
    llm = Bedrock(model_id="mistral.mistral-7b-instruct-v0:2", client=bedrock,
                  model_kwargs={'max_tokens': 500})
    return llm

def get_llama3_llm():
    llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock,
                  model_kwargs={'max_gen_len': 500})
    return llm

def get_amazon_llm():
    llm = Bedrock(model_id="amazon.titan-text-lite-v1", client=bedrock,
                  model_kwargs={'maxTokenCount': 500})
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>
Question: {question}
Assistant:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")

    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Aileron&display=swap');
            .container {
                background-color: #f0f0f0;
                padding: 20px;
                border-radius: 10px;
                display: flex;
                align-items: center;
                max-width: 800px;
                margin: 0 auto;
            }
            .title {
                text-align: center;
                font-family: 'Aileron', sans-serif;
                color: #333;
            }
            .subtitle {
                font-family: 'Aileron', sans-serif;
                color: #555;
            }
            .upload-container {
                padding: 10px;
                border: 2px dashed #ddd;
                border-radius: 10px;
                text-align: center;
                background-color: #fafafa;
            }
            .sidebar-title {
                font-size: 20px;
                font-weight: bold;
                color: #4CAF50;
                margin-bottom: 10px;
            }
            .footer {
                text-align: center;
                margin-top: 20px;
                font-size: 14px;
                color: #888;
            }
        </style>
        <div class="container">
            <h2 class="title">Retrieval Augmented Generation (RAG) using Open-Source Models</h2>
        </div>
    """, unsafe_allow_html=True)

    # st.header("Retrieval Augmented Generation (RAG) using Open-Source Models")
    st.write("<br>", unsafe_allow_html=True)
    
    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.markdown('<div class="sidebar-title">Update Or Create Vector Store:</div>', unsafe_allow_html=True)
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Vector store updated successfully!")
        
        st.markdown('<div class="sidebar-title">Upload PDF files:</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader("", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                with open(os.path.join("data", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success("Files uploaded successfully!")

    if 'llama3_response' not in st.session_state:
        st.session_state.llama3_response = ""

    if 'mistral_response' not in st.session_state:
        st.session_state.mistral_response = ""

    if 'amazon_response' not in st.session_state:
        st.session_state.amazon_response = ""

    with st.expander("Llama3 Output", expanded=True):
        if st.button("Generate Llama3 Output"):
            with st.spinner("Processing..."):
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                llm = get_llama3_llm()
                response = get_response_llm(llm, faiss_index, user_question)
                print(response)
                print("-------------------")
                st.session_state.llama3_response = response
                print(response)
                st.write(response)
        else:
            st.write(st.session_state.llama3_response)

    with st.expander("Mistral Output", expanded=True):
        if st.button("Generate Mistral Output"):
            with st.spinner("Processing..."):
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                llm = get_mistral_llm()
                response = get_response_llm(llm, faiss_index, user_question)
                st.session_state.mistral_response = response
                st.write(st.session_state.mistral_response)
        else:
            st.write(st.session_state.mistral_response)

    with st.expander("Amazon Titan Output", expanded=True):
        if st.button("Generate Amazon Titan Output"):
            with st.spinner("Processing..."):
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                llm = get_amazon_llm()
                response = get_response_llm(llm, faiss_index, user_question)
                st.session_state.amazon_response = response
                st.write(st.session_state.amazon_response)
        else:
            st.write(st.session_state.amazon_response)

    st.markdown('<div class="footer">Made with Love by Sharrr ♥︎</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
