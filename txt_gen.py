!pip install langchain
!pip install openai
!pip install PyPDF2
!pip install faiss-cpu
!pip install tiktoken
!pip install datasets
!pip install gradio
!pip install pydantic

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
reader = PdfReader('/content/drive/MyDrive/llm/48lawsofpower.pdf')

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
        
        
query = "“Can you give me an example from history where the enemy was crushed totallyfrom the book?"
docs = docsearch.similarity_search(query)
chain.run(input_documents=docs, question=query)
