import os
import tempfile
import sys
import tkinter as tk
from tkinter import filedialog  # Add this import

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
import chainlit as cl

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

embeddings = OpenAIEmbeddings(openai_api_key="sk-")
os.environ['OPENAI_API_KEY'] = "sk-"

def process_file(file_path):
    if file_path.endswith(".txt"):
        Loader = TextLoader
    elif file_path.endswith(".pdf"):
        Loader = PyPDFLoader

    loader = Loader(file_path)

    documents = loader.load()
    docs = text_splitter.split_documents(documents)

    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"

    return docs

def get_docsearch(file_path):
    docs = process_file(file_path)

    docsearch = Chroma.from_documents(docs, embeddings)
    return docsearch

def main(file_path):
    docs = process_file(file_path)

    docsearch = get_docsearch(file_path)

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(max_tokens_limit=4097),
    )

    return chain

def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf"), ("Text files", "*.txt")])

    if not file_path:
        print("No file selected. Exiting.")
        sys.exit(1)

    return file_path

if __name__ == "__main__":
    file_path = select_file()
    chain = main(file_path)

    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        response = chain.respond(question)
        print(response["answer"])
