from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

st.title("AskBuddy - AI QnA Bot")
st.markdown("My QnA Bot with LangChain and Google Gemini")

#To Save the Chat - It saves the content in memory
#Execution 2
if 'messages' not in st.session_state:
    st.session_state.messages = []

#Execution 3
for message in st.session_state.messages:
    role = message['role']
    content = message['content']
    st.chat_message(role).markdown(content)

#Execution 1
query = st.chat_input("Ask anthing here")

#Execution 4
if query:
    st.session_state.messages.append({'role': 'user', 'content': query})
    st.chat_message("user").markdown(query)
    response = llm.invoke(query)
    st.chat_message('ai').markdown(response.content)
    st.session_state.messages.append({'role': 'ai', 'content': response.content})
    
#Open Terminal and run: `streamlit run bot.py`

# while True:
#     query = input("User: ")
#     if query.lower() in ['quit', 'exit', 'bye']:
#         print("GoodBye")
#         break
    
#     result = llm.invoke(que)
    # print("AI: ", result.content)
