# AcademAI
(NLP Project – Kahaan Shah & Suyog Joshi)

## About
This is a Llama based, context-aware chatbot that uses academic policy documents from Ashoka University to answer questions. The chatbot is history-aware as well.

## Dependencies
The following python libraries are required for the code to run:
- pypdf
- langchain, langchain-community, langchainhub, langchain-chroma
- nltk
- transformers
- streamlit, streamlit-chat
- llamaapi

## Files and Folders
- [project_docs](project_docs): Contains all the PDFs to be vectorised and used for context
- [vectorisation.py](vectorisation.py): Contains the functions for vectorising docs with specific inputs
- [llama.py](llama.py): Contains the functions for building various chains using LangChain
- [basic_chatbot.py](basic_chatbot.py): A basic terminal based chatbot, adding a --vectorise flag first vectorises according to the parameters specified in the script itself
- [streamlit_trial.py](streamlit_trial.py): Streamlit frontend for the chatbot with a working history
- [presentation.pdf](presentation.pdf): Our presentation for the course that accompanied the project. Details the LangChain architecture and evaluation methods we used

## Execution
After installing all the dependencies first run `python basic_chatbot.py --vectorise` to build the vector database. This may take some time depending on how many files there are in project_docs. After this to launch the streamlit frontend run `streamlit run streamlit_trial.py`. The code has been tested on Windows and MacOS. **Remember to add your own API key in the code.**