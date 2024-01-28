from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from datasets import load_dataset
import os
import gradio as gr
from transformers import pipeline
from datasets import DatasetDict

os.environ["OPENAI_API_KEY"] = "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"

reader = load_dataset("orderlymirror/The_48_Laws_Of_Power")
reader = PdfReader('48lawsofpower.pdf')

raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text
        
        
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

embeddings = OpenAIEmbeddings()

docsearch = FAISS.from_texts(texts, embeddings)

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
        
chain = load_qa_chain(OpenAI(), chain_type="stuff")

def interactive_search(query):
    docs = docsearch.similarity_search(query)
   
    result = chain.run(input_documents=docs, question=query)
    
    return result

iface = gr.Interface(interactive_search, 
                     inputs="text",
                     outputs="text",
                     title="text generator",
                     description="Enter any query from the document 48 laws of power.")
iface.launch()
