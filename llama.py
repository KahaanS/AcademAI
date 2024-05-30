import warnings
warnings.filterwarnings("ignore")

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.evaluation import load_evaluator
from langchain_core.prompts import PromptTemplate

#Get the history aware retriever
def get_history_aware_retriever(retriever, llm):
    
    #This prompt helps to create the intermediate prompt with the history to get the documents
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Use the context provided you. \
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    return history_aware_retriever

#Get the Q-A Chain
def get_qa_chain(llm):
    
    #This prompt helps to create the intermediate prompt with the context to get the answer
    qa_system_prompt = """You are AcademAI, an assistant who answers questions about academic policies at Ashoka University. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. Do not make up an answer if you don't know. \
    Use three sentences maximum and keep the answer as concise as possible. \

    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)   
    return question_answer_chain

#Get the combined chain
def get_rag_chain(retriver, llm):
    
    history_aware_retriever = get_history_aware_retriever(retriver, llm)
    question_answer_chain = get_qa_chain(llm)
    
    #Create the combined chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain) 
    return rag_chain

def get_eval_chain(LLM):
    accuracy_criteria = {
    "accuracy": """
    Score 0: The answer is completely incorrect.
    Score 2: The answer is mostly incorrect.
    Score 5: The answer has some correct information but is mostly incorrect.
    Score 7: The answer is mostly correct but contains minor inaccuracies.
    Score 10: The answer is completely accurate and aligns perfectly with the reference."""
    }

    fstring = """Grade the response based on the following rubric. The response should be graded on the accuracy of the response compared to the expected response. You can grade the response on a scale of 0-10, not just the scores mentioned in the rubric. Be strict.

    Grading Rubric: {criteria}


    DATA:
    ---------
    Question: {input}
    Expected Response: {reference}
    Response: {output}
    ---------
    Write out ONLY the grade you give it. Do NOT write anything else! Print it out in the following format: "accuracy: [NUMBER]", replacing the [NUMBER] field with your scores."""

    prompt = PromptTemplate.from_template(fstring)

    evaluator = load_evaluator(
        "labeled_criteria",
        criteria=accuracy_criteria,
        llm=LLM,
        prompt=prompt,
    )
    
    return evaluator