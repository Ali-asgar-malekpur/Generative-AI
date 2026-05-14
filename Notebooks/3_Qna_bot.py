#LLM
#TOOL - GoogleSearchTool
#AGENT
#MEMORY
#STREAMING
#WEBINTERFACE

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

llm = ChatGroq(model = 'openai/gpt-oss-20b')
search = GoogleSerperAPIWrapper #search is an object that has tool: serach.run
tools = [search.run]
memory = MemorySaver()