import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from llamaapi import LlamaAPI
from langchain_experimental.llms import ChatLlamaAPI
from langchain_core.messages import HumanMessage
from streamlit_chat import message

from llama import get_rag_chain

import asyncio

#Just some streamlit related setup
def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

text_area_id = 'textarea'

scroll_script = f"""
<script>
  var textArea = document.getElementById("{text_area_id}");
  textArea.scrollTop = textArea.scrollHeight;
</script>
"""

#Setup llama variables
API_KEY = ""
LLAMA = LlamaAPI(API_KEY)
LLM = ChatLlamaAPI(client=LLAMA)
PERSIST_DIRECTORY = "chroma"

#Setup docuemnt retriver
EMBEDDING = HuggingFaceEmbeddings()
THRESHOLD = 0.3
vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=EMBEDDING)
#retriever = vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": THRESHOLD})
retriever = vectordb.as_retriever(search_type="mmr")

rag_chain = get_rag_chain(retriever, LLM)

st.title('AcademAI Chatbot')

if 'chat_history_display' not in st.session_state:
    st.session_state.chat_history_display = []
    st.session_state.chat_history = ['']

if 'source' not in st.session_state:
    st.session_state.source = []
#Display chat history


response_container = st.container(height=300)
container = st.container()


with container:
    with st.form(key='my_form', clear_on_submit=True):
        question = st.text_input('Text AcademAI:', '', key='input')
        submit_button = st.form_submit_button(label='Send')

    if question and submit_button:
        ai_msg = rag_chain.invoke({"input": question, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.extend([HumanMessage(content = question), ai_msg["answer"]])
        st.session_state.chat_history_display.append((question, ai_msg["answer"]))
        if ai_msg["context"][0]:
            st.session_state.source.append(ai_msg["context"][0].metadata['source'].split('/')[-1].split('.')[0])
        else:
            st.session_state.source.append['None']
            st.write('Source: None')
        print(st.session_state.chat_history)



with response_container:
    st.write('Chat History:')
    i=0
    for message1, response in st.session_state.chat_history_display:
        message(message1, is_user=True, key=str(i)+'_user',  avatar_style="big-smile")    
        message(response + "\n" + "Source: " + st.session_state.source[i], is_user=False, key=str(i)+'_bot', avatar_style="bottts")
        # st.write('User:', message)
        # st.write('AcademAI:', response)
        # st.write("Source:", st.session_state.source[i])
        i+=1
        st.markdown(scroll_script, unsafe_allow_html=True)