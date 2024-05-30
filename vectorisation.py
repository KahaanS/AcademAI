import warnings
warnings.filterwarnings("ignore")

from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import NLTKTextSplitter
import nltk
import os

def vectorise(dir, persist_dir, chunk_size, chunk_overlap):
    docs = []

    for filename in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, filename)):
            docs.extend(PyPDFLoader(os.path.join(dir, filename)).load())
    
    nltk.download('punkt')
    
    for doc in docs:
        doc.page_content = doc.page_content.replace("\n", " ")
    
    nltk_text_splitter = NLTKTextSplitter(chunk_size=chunk_size, chunk_overlap = chunk_overlap, separator = ' ')
    splits = nltk_text_splitter.split_documents(docs)
    
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=HuggingFaceEmbeddings(),
        persist_directory=persist_dir
    )
    
    print("Finished vectorising")

def vectorise_and_return_db(dir, chunk_size, chunk_overlap):
    
    print("Vectorising, Chunk Size: ", chunk_size, " Chunk Overlap: ", chunk_overlap, " Directory: ", dir)
    
    docs = []

    for filename in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, filename)):
            docs.extend(PyPDFLoader(os.path.join(dir, filename)).load())
    
    nltk.download('punkt')
    
    for doc in docs:
        doc.page_content = doc.page_content.replace("\n", " ")
    
    nltk_text_splitter = NLTKTextSplitter(chunk_size=chunk_size, chunk_overlap = chunk_overlap, separator = ' ')
    splits = nltk_text_splitter.split_documents(docs)
    
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=HuggingFaceEmbeddings(),
    )
    
    print("Finished vectorising")
    return vectordb

