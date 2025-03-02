import os

#pip install pypdf
#export HNSWLIB_NO_NATIVE = 1 The line export HNSWLIB_NO_NATIVE = 1 sets an environment variable HNSWLIB_NO_NATIVE to 1. This environment variable is used to disable the native implementation of HNSW (Hierarchical Navigable Small World) library in favor of the pure Python implementation.

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
import chainlit as cl
from  chainlit.types import AskFileResponse
import os

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

embeddings = OpenAIEmbeddings(openai_api_key="sk-")
os.environ['OPENAI_API_KEY']="sk-"

welcome_message = """Welcome to the QA demo!"""

import tempfile

def process_file(file: AskFileResponse):
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.content)
        temp_file_path = temp_file.name  # Get the path of the temporary file
        loader = Loader(temp_file_path)

    documents = loader.load()
    docs = text_splitter.split_documents(documents)

    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"

    return docs



def get_docsearch(file: AskFileResponse):
    docs = process_file(file)

    # Save data in the user session
    cl.user_session.set("docs", docs)

    # Create a unique namespace for the file

    docsearch = Chroma.from_documents(
        docs, embeddings
    )
    return docsearch


@cl.on_chat_start
async def start():
    
    # Sending an image with the local file path
    await cl.Message(content="Chat with your Documents.").send()

    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # No async implementation in the Pinecone client, fallback to sync
    docsearch = await cl.make_async(get_docsearch)(file)

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=1, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(max_tokens_limit=4097),
    )

    # Let the user know that the system is ready
    msg.content = f"`{file.name}` processed. The system is READY🤖!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])

    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []

    # Get the documents from the user session
    docs = cl.user_session.get("docs")
    metadatas = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in metadatas]

    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = docs[index].page_content
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer, elements=source_elements).send()
        
        
        
'''
This code is an implementation of a question-answering system uses Langchain library to process PDF files and answer questions based on their content. 
Here's a breakdown of the code:

The necessary modules are imported, including Langchain modules for loading documents, splitting text, creating embeddings, and building a vector database.
The RecursiveCharacterTextSplitter is used to split the text from the documents into smaller chunks of 1000 characters each, with an overlap of 100 characters between chunks.
The OpenAIEmbeddings class is used to embed the text chunks into vector space using the OpenAI embedding API.
The Chroma class is used to create a vector database that stores the embedded text chunks and allows for efficient retrieval of relevant chunks based on vector similarity.
The RetrievalQAWithSourcesChain class is used to create a question-answering chain that first retrieves relevant text chunks from the vector database, and then generates an answer to the user's question using the ChatOpenAI class.
The chainlit module is used to create a user interface for the question-answering system.
The process_file function takes a file as input, determines whether it is a text or PDF file, and then loads the file using the appropriate loader. The text is then split into smaller chunks using the RecursiveCharacterTextSplitter.
The get_docsearch function processes the file using the process_file function and creates a vector database using the Chroma class.
The start function is called when the chat session starts. It prompts the user to upload a file, processes the file, creates a vector database, and then creates a question-answering chain using the RetrievalQAWithSourcesChain class.
The main function is called when a user sends a message. It uses the RetrievalQAWithSourcesChain class to generate an answer to the user's question and then sends the answer back to the user.
The source_elements list is used to store the text elements that are referenced in the answer. These elements are then displayed to the user along with the answer.
Overall, this code creates a system that can answer questions based on the content of PDF files, using natural language processing and vector embeddings to understand and retrieve relevant information.
'''