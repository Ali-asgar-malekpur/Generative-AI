from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

llm = ChatGroq(model = 'openai/gpt-oss-20b', streaming = True)
search = GoogleSerperAPIWrapper()
tools = [search.run]

# Very Important.
# The server will be refreshed each time if this is not defined.
# Thus by the earlier asked questions won't be saved.
# So we create a session that will store the current memory and pass it to agent
if 'memory' not in st.session_state:
    st.session_state.memory = MemorySaver() #memory is a variable in st.session_state
    st.session_state.history = [] #history is a variable in st.session_state

agent = create_agent(
    model = llm,
    tools = tools,
    checkpointer = st.session_state.memory,
    system_prompt = 'You are a amazing AI agent and can search on Google as well'
)

# Building Web Interface
st.subheader('QuickAnswer - Answer at the speed of Thought!')

#To show all asked/responded user/ai messages in web:
for message in st.session_state.history:
    role = message['role']
    content = message['content']
    st.chat_message(role).markdown(content)

query = st.chat_input('Ask Anything')
if query:
    st.chat_message('user').markdown(query) # This will show the query of user in web as user asked question
    st.session_state.history.append({'role':'user', 'content':query})

    response = agent.stream(
        {'messages':[{'role':'user', 'content':query}]},
        {'configurable':{'thread_id': 'Ali_constant'}},
        stream_mode = 'messages'
    )

    #To have typing effect:
    ai_container = st.chat_message('ai')
    with ai_container:
        space = st.empty()

        message = ''
        for chunk in response:
            message = message + chunk[0].content
            space.write(message)
        
    # answer = response['messages'][-1].content
    # st.chat_message('ai').markdown(answer)

        st.session_state.history.append({'role':'AI', 'content':message})