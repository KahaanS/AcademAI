import warnings
warnings.filterwarnings("ignore")

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from llamaapi import LlamaAPI
from langchain_experimental.llms import ChatLlamaAPI
from langchain_core.messages import HumanMessage
from llama import get_rag_chain

from vectorisation import vectorise
import sys

CHUNK_SIZE = 1100
CHUNK_OVERLAP = 300
DOC_DIR = "project_docs"
PERSIST_DIRECTORY = "chroma"

if len(sys.argv) > 1 and sys.argv[1] == "--vectorise":
    print("Vectorising")
    vectorise(DOC_DIR, PERSIST_DIRECTORY, CHUNK_SIZE, CHUNK_OVERLAP)

#Setup llama variables
API_KEY = ""
LLAMA = LlamaAPI(API_KEY)
LLM = ChatLlamaAPI(client=LLAMA)

#Setup docuemnt retriver
EMBEDDING = HuggingFaceEmbeddings()
vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=EMBEDDING)
#retriever = vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.3})
retriever = vectordb.as_retriever(search_type="mmr")

rag_chain = get_rag_chain(retriever, LLM)

question = ""
chat_history = [""]
while (True):
  question = input("Text AcademAI: ")
  if question == "end chat rn":
    break
  ai_msg = rag_chain.invoke({"input": question, "chat_history": chat_history})
  chat_history.extend([HumanMessage(content = question), ai_msg["answer"]])
  answer = ai_msg["answer"]
  source = [ai_msg["context"]]
  print(f"\n\nAcademAI: {answer} \n\nSource: {source}\n\n")